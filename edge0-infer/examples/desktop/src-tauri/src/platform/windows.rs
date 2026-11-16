use std::{
    os::windows::{io::AsRawHandle, process::CommandExt},
    path::PathBuf,
    process::{Child, Command},
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::{
        Console::{
            AllocConsole, GenerateConsoleCtrlEvent, GetConsoleProcessList, GetConsoleWindow,
            CTRL_BREAK_EVENT,
        },
        JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        },
        SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX},
        Threading::{
            OpenProcess, QueryFullProcessImageNameW, TerminateProcess, CREATE_NEW_PROCESS_GROUP,
            PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
        },
    },
    UI::WindowsAndMessaging::{ShowWindow, SW_HIDE},
};
/// The job the engine is kept in, when one could be made.  `None` means the app could not put the engine into a job
/// (see `attach`) and tracks it by pid instead, which is what `serve-pid.json`, `graceful` and `force` already do.
pub struct Lifetime(Option<HANDLE>);
// The handle is owned, is never aliased, and is only closed when its owner is dropped.
unsafe impl Send for Lifetime {}
impl Drop for Lifetime {
    fn drop(&mut self) {
        if let Some(handle) = self.0 {
            unsafe {
                CloseHandle(handle);
            }
        }
    }
}
pub fn default_device() -> &'static str {
    "cpu"
}
pub fn configure(command: &mut Command) -> Result<(), String> {
    unsafe {
        let mut processes = [0_u32; 1];
        // A release GUI has no console. Create a hidden host console so the
        // child can inherit it while becoming a distinct, signallable group.
        // CREATE_NEW_CONSOLE would cause CREATE_NEW_PROCESS_GROUP to be ignored.
        // Best effort: without a console `graceful` has nothing to signal and the stop path falls back to
        // TerminateProcess, so a host that refuses the console (a confined launcher) must not fail the load.
        if GetConsoleProcessList(processes.as_mut_ptr(), 1) == 0 && AllocConsole() != 0 {
            ShowWindow(GetConsoleWindow(), SW_HIDE);
        }
    }
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    Ok(())
}
pub fn attach(child: &Child) -> Result<Lifetime, String> {
    // Best effort.  A job gives the engine "killed when the app dies" for free, but the app can only put a process
    // into one when the process is not already inside a job that forbids nesting: AssignProcessToJobObject then
    // fails with ERROR_ACCESS_DENIED (os error 5), which is what a launcher that itself runs processes in a job
    // produces - an agent/sandbox shell, a CI runner, or any tool that starts this app inside its own job.  That is
    // not a reason to refuse to load the model: stopping the engine does not depend on the job (serve-pid.json
    // records the pid, and `graceful`/`force` act on it), so the only thing given up is the automatic kill when the
    // app is killed outright.
    unsafe {
        let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if handle.is_null() {
            return Ok(Lifetime(None));
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let configured = SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of_val(&info) as u32,
        ) != 0;
        if !configured || AssignProcessToJobObject(handle, child.as_raw_handle().cast()) == 0 {
            CloseHandle(handle);
            return Ok(Lifetime(None));
        }
        Ok(Lifetime(Some(handle)))
    }
}
pub fn graceful(pid: u32) -> Result<(), String> {
    // Only the child group receives CTRL_BREAK; the GUI host remains attached
    // to the shared hidden console and is outside that group.
    if unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) } != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().to_string())
    }
}
pub fn executable(pid: u32) -> Option<PathBuf> {
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut path = vec![0_u16; 32768];
        let mut size = path.len() as u32;
        let ok = QueryFullProcessImageNameW(handle, 0, path.as_mut_ptr(), &mut size);
        CloseHandle(handle);
        (ok != 0).then(|| PathBuf::from(String::from_utf16_lossy(&path[..size as usize])))
    }
}
pub fn force(pid: u32) {
    unsafe {
        let handle = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if !handle.is_null() {
            TerminateProcess(handle, 1);
            CloseHandle(handle);
        }
    }
}
/// The system commit limit (RAM visible to Windows plus the page file) and how much of it is still free.  An
/// allocation past the limit fails even when RAM looks free, and a GPU that shares system memory charges its VRAM to it.
pub fn commit_memory() -> Option<crate::service::CommitMemory> {
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(crate::service::CommitMemory {
        limit: status.ullTotalPageFile,
        available: status.ullAvailPageFile,
    })
}
