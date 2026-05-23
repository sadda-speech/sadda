//! F10 — single-writer lock integration tests. Validates that the
//! `.sadda-lock` file is acquired on `Project::open` / `create` and
//! released on `Drop`, that double-acquire from a live process
//! errors, and that a stale lock from a dead PID is cleared.

use std::path::PathBuf;

use sadda_engine::{EngineError, Project};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_f10_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

#[test]
fn create_writes_lockfile() {
    let root = unique_dir("create_lock");
    let _ = std::fs::remove_dir_all(&root);

    let project = Project::create(&root, "p").unwrap();
    assert!(root.join(".sadda-lock").is_file());
    drop(project);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn drop_releases_lockfile() {
    let root = unique_dir("drop_releases");
    let _ = std::fs::remove_dir_all(&root);

    {
        let _project = Project::create(&root, "p").unwrap();
        assert!(root.join(".sadda-lock").is_file());
    } // drop here
    assert!(
        !root.join(".sadda-lock").exists(),
        "lockfile should be deleted on Drop"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn open_takes_over_lock_from_same_pid() {
    // The GUI's typical re-open-after-close flow lands here: we
    // dropped the old Project (lock released), then open again
    // (lock re-acquired). And even if we forgot to drop, the
    // same-PID branch takes ownership silently.
    let root = unique_dir("same_pid_takeover");
    let _ = std::fs::remove_dir_all(&root);

    let p1 = Project::create(&root, "p").unwrap();
    // Forge a stale lockfile-style situation: keep p1 alive (so
    // lock is held), then call open() in the same process.
    let p2 = Project::open(&root).unwrap();
    drop(p2);
    drop(p1);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn stale_lock_from_dead_pid_is_cleared() {
    let root = unique_dir("stale_lock");
    let _ = std::fs::remove_dir_all(&root);
    let project = Project::create(&root, "p").unwrap();
    drop(project);

    // Write a fake lockfile claiming a definitely-dead PID on the
    // current hostname. PID 0 is special on UNIX (never assignable
    // to a real process); MaxValue is "extremely unlikely to
    // exist" on any OS.
    let hostname = std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let fake_pid = u32::MAX;
    let lockfile = root.join(".sadda-lock");
    std::fs::write(
        &lockfile,
        format!("pid = {fake_pid}\nhostname = {hostname:?}\nacquired_at = \"epoch+0s\"\n"),
    )
    .unwrap();

    // Reopen should clear the stale lock and succeed.
    let project = Project::open(&root).unwrap();
    drop(project);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn locked_by_different_pid_errors() {
    let root = unique_dir("different_pid_locked");
    let _ = std::fs::remove_dir_all(&root);
    let project = Project::create(&root, "p").unwrap();
    drop(project);

    // Write a lockfile claiming a different PID that is definitely
    // live — our parent process. `getppid()` on UNIX; on Windows we
    // approximate by the current PID + 1 (likely live as well, but
    // unreliable — UNIX-only test).
    #[cfg(unix)]
    let live_pid: u32 = unsafe { libc::getppid() } as u32;
    #[cfg(not(unix))]
    let live_pid: u32 = std::process::id(); // same-PID branch: takeover, not error

    let hostname = std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    std::fs::write(
        root.join(".sadda-lock"),
        format!("pid = {live_pid}\nhostname = {hostname:?}\nacquired_at = \"epoch+0s\"\n",),
    )
    .unwrap();

    #[cfg(unix)]
    {
        let err = Project::open(&root).unwrap_err();
        match err {
            EngineError::ProjectLocked {
                holder_pid,
                hostname: h,
                ..
            } => {
                assert_eq!(holder_pid, live_pid);
                assert_eq!(h, hostname);
            }
            other => panic!("expected ProjectLocked, got {other:?}"),
        }
    }
    #[cfg(not(unix))]
    {
        // On non-UNIX, the same-PID branch takes ownership; just
        // confirm open succeeds.
        let _project = Project::open(&root).unwrap();
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
extern crate libc;
