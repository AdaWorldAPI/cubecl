use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{self, AtomicUsize},
        mpsc,
    },
};

use crossbeam_utils::CachePadded;

use crate::compute::{
    affinity::{CoreId, get_active_cores},
    threadpool::{ThreadTask, compute_task::ComputeTask, scheduler::Worker},
};

pub struct DispatcherScheduler {
    cores: Vec<CoreId>,
    tx: Vec<mpsc::Sender<ComputeTask>>,
    lens: Vec<Arc<CachePadded<AtomicUsize>>>,
}

impl Default for DispatcherScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl DispatcherScheduler {
    pub fn new() -> Self {
        let cores: Vec<_> = get_active_cores().collect();
        let mut scheduler = Self {
            cores,
            tx: Vec::new(),
            lens: Vec::new(),
        };
        // Start with one worker per active core; the pool grows on demand when a
        // parallel cube needs more units than we currently have workers.
        let cores = scheduler.cores.len();
        scheduler.ensure_workers(cores);
        scheduler
    }

    /// Spawns workers until at least `n` exist. Overflow workers past the core
    /// count round-robin over the active cores, so a cube with more units than
    /// cores still gets one thread per unit — required so the `sync_cube` spin
    /// barrier never queues two units of the same cube behind each other.
    ///
    /// Overflow workers are never shrunk, so they must stay reserved for
    /// barrier units: [`select_target`] keeps ordinary work on the first
    /// `cores.len()` workers.
    pub fn ensure_workers(&mut self, n: usize) {
        while self.tx.len() < n {
            let core_id = self.cores[self.tx.len() % self.cores.len()];
            let (worker, tx, len) = DispatcherWorker::new();
            worker.spawn_thread(core_id);
            self.tx.push(tx);
            self.lens.push(len);
        }
    }

    pub fn send(&mut self, index: usize, task: ComputeTask) {
        let needs_parallelism = task.pliron_engine.requirements().needs_parallelism;
        let core_workers = self.cores.len().min(self.lens.len());
        let target = select_target(needs_parallelism, index, core_workers, |i| {
            self.lens[i].load(atomic::Ordering::Relaxed)
        });
        let _ = self.tx[target].send(task);
        self.lens[target].fetch_add(1, atomic::Ordering::Relaxed);
    }
}

/// Chooses the worker a unit is dispatched to.
///
/// * Barrier units (`needs_parallelism`) are affine: unit `index` always runs
///   on worker `index`, one worker per unit, so the `sync_cube` spin barrier
///   never finds two units of one cube queued behind each other. The caller
///   grew the pool via [`DispatcherScheduler::ensure_workers`], so `index` is
///   in range.
/// * Ordinary units load-balance onto the least-loaded of the first
///   `core_workers` workers, lowest index on a tie. `index` is a unit
///   position that can exceed the worker count, so it is never used here.
///   Overflow workers, spawned for a barrier cube wider than the core count,
///   are excluded: the pool never shrinks, so counting them would let one
///   wide barrier launch turn every later ordinary launch into more threads
///   than cores.
///
/// `len_of(i)` reports worker `i`'s queued-task count. Kept free of any task
/// or channel type so the policy is testable without compiling a kernel.
fn select_target(
    needs_parallelism: bool,
    index: usize,
    core_workers: usize,
    len_of: impl Fn(usize) -> usize,
) -> usize {
    if needs_parallelism {
        return index;
    }
    let mut best = 0;
    let mut min_value = len_of(0);
    for i in 1..core_workers {
        let len = len_of(i);
        if len < min_value {
            best = i;
            min_value = len;
        }
    }
    best
}

pub struct DispatcherWorker {
    rx: mpsc::Receiver<ComputeTask>,
    aside: VecDeque<ComputeTask>,
    len: Arc<CachePadded<AtomicUsize>>,
}

impl DispatcherWorker {
    fn new() -> (
        Self,
        mpsc::Sender<ComputeTask>,
        Arc<CachePadded<AtomicUsize>>,
    ) {
        let (tx, rx) = mpsc::channel();
        let aside = VecDeque::with_capacity(4);
        let len = Arc::new(CachePadded::new(AtomicUsize::new(0)));
        let worker = Self {
            rx,
            aside,
            len: len.clone(),
        };
        (worker, tx, len)
    }
}

/// How long an idle worker polls `try_recv`, yielding the CPU between
/// misses, before parking in the blocking `recv`. A parked worker costs a
/// futex wake per launch, and a barrier kernel runs at the latency of its
/// last-woken unit, while the budget caps what an actually-idle pool burns.
const IDLE_POLL: std::time::Duration = std::time::Duration::from_micros(200);

impl Worker for DispatcherWorker {
    fn work(mut self) {
        loop {
            if self.aside.is_empty() {
                let mut received = None;
                let poll_start = std::time::Instant::now();
                while poll_start.elapsed() < IDLE_POLL {
                    match self.rx.try_recv() {
                        Ok(task) => {
                            received = Some(task);
                            break;
                        }
                        // The workers cover every logical CPU, so polling in earnest
                        // starves the client and any still-running units.
                        Err(_) => std::thread::yield_now(),
                    }
                }
                let task = match received {
                    Some(task) => Ok(task),
                    None => self.rx.recv(),
                };
                if let Ok(mut task) = task {
                    if task.is_ready() {
                        task.compute();
                        self.len.fetch_sub(1, atomic::Ordering::Relaxed);
                    } else {
                        self.aside.push_back(task);
                    }
                }
            } else if self.aside.len() < 4 {
                let task = self.rx.try_recv();
                if let Ok(task) = task {
                    self.aside.push_back(task);
                }
            }
            self.aside.retain_mut(|elem| {
                if elem.is_ready() {
                    elem.compute();
                    self.len.fetch_sub(1, atomic::Ordering::Relaxed);
                    false
                } else {
                    std::hint::spin_loop();
                    true
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::select_target;

    const CORES: usize = 8;
    const WIDE_BARRIER: usize = 64;

    /// A barrier unit runs on its own worker, however the load looks. Worker
    /// 0 is the least loaded here, so a policy that load-balanced barrier
    /// units would pick it instead.
    #[test]
    fn barrier_units_stay_on_their_own_worker() {
        let lens: Vec<usize> = (0..WIDE_BARRIER).map(|i| i + 1).collect();
        for index in 0..WIDE_BARRIER {
            assert_eq!(
                select_target(true, index, CORES, |i| lens[i]),
                index,
                "barrier unit {index} must not move"
            );
        }
    }

    /// The silence half: an ordinary unit is never pinned to the worker its
    /// position names. Unit 5 goes to the least-loaded core worker, 2.
    #[test]
    fn ordinary_units_ignore_their_unit_position() {
        let lens = [4, 4, 0, 4, 4, 4, 4, 4];
        assert_eq!(select_target(false, 5, CORES, |i| lens[i]), 2);
    }

    /// After a wide barrier launch grew the pool to 64 workers, ordinary work
    /// must stay on the 8 core workers. The overflow workers are made strictly
    /// idler than every core worker, so a scan over the whole pool would pick
    /// one of them.
    #[test]
    fn ordinary_units_never_land_on_overflow_workers() {
        let lens: Vec<usize> = (0..WIDE_BARRIER)
            .map(|i| if i < CORES { 3 } else { 0 })
            .collect();
        assert!(
            lens[CORES..]
                .iter()
                .all(|&o| lens[..CORES].iter().all(|&c| o < c)),
            "fixture: every overflow worker must be idler than every core worker"
        );
        for index in 0..WIDE_BARRIER {
            let target = select_target(false, index, CORES, |i| lens[i]);
            assert!(
                target < CORES,
                "unit {index} dispatched to overflow worker {target}"
            );
        }
    }

    /// Least loaded wins; the lowest index breaks ties.
    #[test]
    fn ordinary_units_pick_the_least_loaded_core_worker() {
        let lens = [2, 1, 3, 1, 5, 9, 1, 2];
        assert_eq!(select_target(false, 0, CORES, |i| lens[i]), 1);
    }
}
