use std::{
    path::{Path, PathBuf},
    process::{Child, Command},
};
pub struct Lifetime;
pub fn default_device() -> &'static str {
    "metal"
}
pub fn configure(_: &mut Command) -> Result<(), String> {
    Ok(())
}
pub fn attach(_: &Child) -> Result<Lifetime, String> {
    Ok(Lifetime)
}
pub fn graceful(pid: u32) -> Result<(), String> {
    if unsafe { libc::kill(pid as i32, libc::SIGTERM) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().to_string())
    }
}
pub fn executable(pid: u32) -> Option<PathBuf> {
    let mut buffer = vec![0_u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    let len =
        unsafe { libc::proc_pidpath(pid as i32, buffer.as_mut_ptr().cast(), buffer.len() as u32) };
    if len <= 0 {
        return None;
    }
    let end = buffer.iter().position(|b| *b == 0).unwrap_or(len as usize);
    use std::os::unix::ffi::OsStrExt;
    Some(Path::new(std::ffi::OsStr::from_bytes(&buffer[..end])).to_path_buf())
}
pub fn force(pid: u32) {
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}
/// macOS grows its swap on demand and has no fixed commit limit to run into.
pub fn commit_memory() -> Option<crate::service::CommitMemory> {
    None
}

// AppKit panels must run on the application's main thread. The command routes
// this call through AppHandle::run_on_main_thread and waits off the UI thread.
pub fn choose_path(directory: bool) -> Result<Option<String>, String> {
    use std::ffi::{c_char, c_void, CStr};
    type Object = *mut c_void;
    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Object;
        fn sel_registerName(name: *const c_char) -> Object;
        fn objc_msgSend();
    }
    unsafe fn selector(name: &'static CStr) -> Object {
        sel_registerName(name.as_ptr())
    }
    unsafe fn object(receiver: Object, name: &'static CStr) -> Object {
        let send: unsafe extern "C" fn(Object, Object) -> Object =
            std::mem::transmute(objc_msgSend as *const ());
        send(receiver, selector(name))
    }
    unsafe fn boolean(receiver: Object, name: &'static CStr, value: bool) {
        let send: unsafe extern "C" fn(Object, Object, i8) =
            std::mem::transmute(objc_msgSend as *const ());
        send(receiver, selector(name), i8::from(value));
    }
    unsafe {
        let panel = object(objc_getClass(c"NSOpenPanel".as_ptr()), c"openPanel");
        if panel.is_null() {
            return Err("Cannot create the system file chooser".into());
        }
        boolean(panel, c"setCanChooseDirectories:", directory);
        boolean(panel, c"setCanChooseFiles:", !directory);
        boolean(panel, c"setAllowsMultipleSelection:", false);
        boolean(panel, c"setCanCreateDirectories:", false);
        let application = object(
            objc_getClass(c"NSApplication".as_ptr()),
            c"sharedApplication",
        );
        boolean(application, c"activateIgnoringOtherApps:", true);
        let run: unsafe extern "C" fn(Object, Object) -> isize =
            std::mem::transmute(objc_msgSend as *const ());
        if run(panel, selector(c"runModal")) != 1 {
            return Ok(None);
        }
        let url = object(panel, c"URL");
        if url.is_null() {
            return Ok(None);
        }
        let path = object(url, c"path");
        let utf8 = object(path, c"UTF8String");
        if utf8.is_null() {
            return Err("Cannot read the selected path".into());
        }
        CStr::from_ptr(utf8.cast())
            .to_str()
            .map(|path| Some(path.to_owned()))
            .map_err(|_| "The selected path is not UTF-8".into())
    }
}
