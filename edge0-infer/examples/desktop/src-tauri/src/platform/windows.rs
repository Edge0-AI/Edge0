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
        Threading::{
            OpenProcess, QueryFullProcessImageNameW, TerminateProcess, CREATE_NEW_PROCESS_GROUP,
            PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE,
        },
    },
    UI::WindowsAndMessaging::{ShowWindow, SW_HIDE},
};
pub struct Lifetime(HANDLE);
// The handle is owned, is never aliased, and is only closed when its owner is dropped.
unsafe impl Send for Lifetime {}
impl Drop for Lifetime {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
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
        if GetConsoleProcessList(processes.as_mut_ptr(), 1) == 0 {
            if AllocConsole() == 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            ShowWindow(GetConsoleWindow(), SW_HIDE);
        }
    }
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    Ok(())
}
pub fn attach(child: &Child) -> Result<Lifetime, String> {
    unsafe {
        let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if handle.is_null() {
            return Err(std::io::Error::last_os_error().to_string());
        }
        let job = Lifetime(handle);
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            std::mem::size_of_val(&info) as u32,
        ) == 0
            || AssignProcessToJobObject(handle, child.as_raw_handle().cast()) == 0
        {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(job)
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
