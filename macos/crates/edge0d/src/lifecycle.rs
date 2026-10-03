//! Cascade-stop watch, startup orphan sweep, and App Nap prevention.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use libloading::{Library, Symbol};

use crate::app::AppRef;

/// Under menu-agent: host app pid gone → reclaim workers and exit.
pub fn spawn_owner_monitor(app: AppRef, stop: Arc<AtomicBool>) {
    let Some(owner) = app.owner_pid else { return };
    if owner == 0 {
        return;
    }
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(200));
            let alive = unsafe { libc::kill(owner as i32, 0) } == 0;
            if !alive {
                eprintln!("edge0d: host app (pid {owner}) gone → cascade stop");
                app.pool.stop_all("cascade");
                app.bus.flush_persist();
                app.clear_daemon_json();
                std::process::exit(0);
            }
        }
    });
}

/// Kill only processes named `edge0-engine` whose path is in this daemon's directory.
pub fn orphan_sweep() -> Vec<u32> {
    let mut killed = Vec::new();
    let exe_dir = match std::env::current_exe() {
        Ok(p) => p.parent().map(|d| d.to_path_buf()).unwrap_or_default(),
        Err(_) => return killed,
    };
    let mut pids = vec![0i32; 4096];
    let sz = std::mem::size_of_val(pids.as_slice());
    let n = unsafe { libc::proc_listallpids(pids.as_mut_ptr() as *mut libc::c_void, sz as i32) };
    if n <= 0 {
        return killed;
    }
    pids.truncate(n as usize / 4);
    for pid in pids {
        if pid <= 1 {
            continue;
        }
        let mut buf = [0i8; 1024];
        let len = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr() as *mut libc::c_void, 1024) };
        if len <= 0 {
            continue;
        }
        let path = String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u8, len as usize)
        })
        .into_owned();
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name == "edge0-engine" && std::path::Path::new(&path).parent() == Some(exe_dir.as_path())
        {
            if unsafe { libc::kill(pid, libc::SIGKILL) } == 0 {
                killed.push(pid as u32);
            }
        }
    }
    killed
}

/// Hold an activity assertion for the process lifetime. On arm64, `objc_msgSend` must be
/// called through a typed function pointer matching the selector arity (variadic SIGSEGVs).
pub fn begin_activity_assertion() -> bool {
    use std::ffi::c_void;
    type Id = *mut c_void;
    type Sel = *mut c_void;
    type Msg0 = unsafe extern "C" fn(Id, Sel) -> Id;
    type MsgStr = unsafe extern "C" fn(Id, Sel, *const i8) -> Id;
    type MsgAct = unsafe extern "C" fn(Id, Sel, u64, Id) -> Id;

    let Ok(lib) = (unsafe { Library::new("/usr/lib/libobjc.A.dylib") }) else {
        return false;
    };
    let Ok(get_class): Result<Symbol<unsafe extern "C" fn(*const i8) -> Id>, _> =
        (unsafe { lib.get(b"objc_getClass\0") })
    else {
        return false;
    };
    let Ok(sel_reg): Result<Symbol<unsafe extern "C" fn(*const i8) -> Sel>, _> =
        (unsafe { lib.get(b"sel_registerName\0") })
    else {
        return false;
    };
    let Ok(msg): Result<Symbol<Msg0>, _> = (unsafe { lib.get(b"objc_msgSend\0") }) else {
        return false;
    };
    let msg = *msg;
    unsafe {
        let info_cls = get_class(c"NSProcessInfo".as_ptr());
        if info_cls.is_null() {
            return false;
        }
        let process_info_sel = sel_reg(c"processInfo".as_ptr());
        let info = msg(info_cls, process_info_sel);
        let str_cls = get_class(c"NSString".as_ptr());
        if str_cls.is_null() || info.is_null() {
            return false;
        }
        let sws_sel = sel_reg(c"stringWithUTF8String:".as_ptr());
        let reason_c =
            std::ffi::CString::new("edge0d resident service: disable App Nap throttling").unwrap();
        let sws: MsgStr = std::mem::transmute::<Msg0, MsgStr>(msg);
        let reason_str = sws(str_cls, sws_sel, reason_c.as_ptr());
        let begin_sel = sel_reg(c"beginActivityWithOptions:reason:".as_ptr());
        let begin: MsgAct = std::mem::transmute::<Msg0, MsgAct>(msg);
        // NSActivityUserInitiatedAllowingIdleSystemSleep = 0x00FFFFFF
        let activity = begin(info, begin_sel, 0x00FFFFFFu64, reason_str);
        if activity.is_null() {
            return false;
        }
        // Retain forever: a leaked handle keeps the assertion until process exit.
        let _ = activity;
        true
    }
}
