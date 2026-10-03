//! Shared `dlopen` + symbol resolve + ABI handshake for daemon and worker.

use std::path::Path;

use libloading::{Library, Symbol};

use crate::version::E0_ABI_VERSION;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct E0TokenOutC {
    pub size: u32,
    pub token: i32,
    pub flags: u32,
}

pub type FnAbiVersion = extern "C" fn() -> u32;
pub type FnEngineCreate = unsafe extern "C" fn(*const i8, *const i8, *const i8) -> *mut c_void;
pub type FnGenerateBegin =
    unsafe extern "C" fn(*mut c_void, *const i32, u32, *const i8) -> *mut c_void;
pub type FnNext = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut E0TokenOutC) -> i32;
pub type FnCancel = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
pub type FnStats = unsafe extern "C" fn(*mut c_void, *mut i8, usize, *mut usize) -> i32;
pub type FnTurnReset = unsafe extern "C" fn(*mut c_void) -> i32;
pub type FnLastError = unsafe extern "C" fn(*mut c_void, *mut i8, usize, *mut usize) -> i32;
pub type FnEngineDestroy = unsafe extern "C" fn(*mut c_void);
pub type FnTokCreate = unsafe extern "C" fn(*const i8) -> *mut c_void;
pub type FnTokCall =
    unsafe extern "C" fn(*mut c_void, *const i8, *mut i8, usize, *mut usize) -> i32;
pub type FnTokDestroy = unsafe extern "C" fn(*mut c_void);

pub use std::ffi::c_void;

pub const E0_OK: i32 = 0;
pub const E0_NOTIFIED: i32 = 1;
pub const E0_WOULD_BLOCK: i32 = 2;

/// Report what was actually loaded; unknown basenames are passed through, not classified.
pub fn engine_backend(path: &Path) -> String {
    let base = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    if base.contains("native") {
        "native".into()
    } else if base.contains("replay") {
        "replay".into()
    } else {
        base
    }
}

/// Version mismatch refuses to load; missing symbols (including `e0_tok_decode`) fail load.
pub struct EngineLib {
    _lib: Library,
    pub path: std::path::PathBuf,
    pub abi_version: FnAbiVersion,
    pub engine_create: FnEngineCreate,
    pub generate_begin: FnGenerateBegin,
    pub next: FnNext,
    pub cancel: FnCancel,
    pub stats: FnStats,
    pub turn_reset: FnTurnReset,
    pub last_error: FnLastError,
    pub engine_destroy: FnEngineDestroy,
    pub tok_create: FnTokCreate,
    pub tok_tokenize: FnTokCall,
    pub tok_apply_template: FnTokCall,
    pub tok_decode: FnTokCall,
    pub tok_destroy: FnTokDestroy,
}

// dyld is thread-safe; function pointers are read-only addresses.
unsafe impl Send for EngineLib {}
unsafe impl Sync for EngineLib {}

macro_rules! sym {
    ($lib:expr, $ty:ty, $name:literal) => {{
        let s: Symbol<$ty> = (unsafe { $lib.get($name.as_bytes()) })
            .map_err(|e| format!("symbol {} missing or wrong type: {e}", $name))?;
        *s
    }};
}

impl EngineLib {
    pub fn backend(&self) -> String {
        engine_backend(&self.path)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }
            .map_err(|e| format!("dlopen {} failed: {e}", path.display()))?;
        let abi_version: FnAbiVersion = sym!(lib, FnAbiVersion, "e0_abi_version");
        let got = abi_version();
        if got != E0_ABI_VERSION {
            return Err(format!(
                "ABI handshake failed: library reports {got}, daemon compiled with E0_ABI_VERSION={E0_ABI_VERSION} — refusing to load"
            ));
        }
        let e = Self {
            path: path.to_path_buf(),
            abi_version,
            engine_create: sym!(lib, FnEngineCreate, "e0_engine_create"),
            generate_begin: sym!(lib, FnGenerateBegin, "e0_generate_begin"),
            next: sym!(lib, FnNext, "e0_next"),
            cancel: sym!(lib, FnCancel, "e0_cancel"),
            stats: sym!(lib, FnStats, "e0_stats"),
            turn_reset: sym!(lib, FnTurnReset, "e0_turn_reset"),
            last_error: sym!(lib, FnLastError, "e0_last_error"),
            engine_destroy: sym!(lib, FnEngineDestroy, "e0_engine_destroy"),
            tok_create: sym!(lib, FnTokCreate, "e0_tok_create"),
            tok_tokenize: sym!(lib, FnTokCall, "e0_tok_tokenize"),
            tok_apply_template: sym!(lib, FnTokCall, "e0_tok_apply_template"),
            tok_decode: sym!(lib, FnTokCall, "e0_tok_decode"),
            tok_destroy: sym!(lib, FnTokDestroy, "e0_tok_destroy"),
            _lib: lib,
        };
        Ok(e)
    }

    /// Two-probe JSON out-param wrapper (returns an owned String).
    ///
    /// # Safety
    /// `f` must report the required length when `buf==NULL` and write exactly `*written` bytes
    /// into a sufficient buffer, capturing only state valid for this call.
    pub unsafe fn call_json(
        &self,
        f: impl Fn(*mut i8, usize, *mut usize) -> i32,
    ) -> Result<String, i32> {
        let mut written = 0usize;
        let st = f(std::ptr::null_mut(), 0, &mut written);
        if st != 0 || written == 0 {
            return Err(st);
        }
        let mut buf = vec![0u8; written];
        let st = f(buf.as_mut_ptr() as *mut i8, written, &mut written);
        if st != 0 {
            return Err(st);
        }
        String::from_utf8(buf).map_err(|_| -1)
    }
}

/// Per-tier tokenizer handle on the daemon; independent of worker keep_alive.
/// Pointers are only touched inside this wrapper, under a Mutex at the call site.
pub struct TokHandle {
    pub lib: std::sync::Arc<EngineLib>,
    ptr: *mut c_void,
}
unsafe impl Send for TokHandle {}

impl TokHandle {
    pub fn open(lib: std::sync::Arc<EngineLib>, model_dir: &Path) -> Result<Self, String> {
        let c =
            std::ffi::CString::new(model_dir.display().to_string()).map_err(|e| e.to_string())?;
        let ptr = unsafe { (lib.tok_create)(c.as_ptr()) };
        if ptr.is_null() {
            return Err(format!(
                "e0_tok_create({}) returned NULL (tokenizer files missing or unreadable)",
                model_dir.display()
            ));
        }
        Ok(Self { lib, ptr })
    }
    fn call(&self, f: FnTokCall, input: &str) -> Result<String, i32> {
        let c = match std::ffi::CString::new(input) {
            Ok(c) => c,
            Err(_) => return Err(-1),
        };
        let lib = &self.lib;
        let p = self.ptr;
        unsafe { lib.call_json(|b, l, w| f(p, c.as_ptr(), b, l, w)) }
    }
    pub fn tokenize(&self, text_json: &str) -> Result<String, i32> {
        self.call(self.lib.tok_tokenize, text_json)
    }
    pub fn apply_template(&self, messages_json: &str) -> Result<String, i32> {
        self.call(self.lib.tok_apply_template, messages_json)
    }
    pub fn decode(&self, tokens_json: &str) -> Result<String, i32> {
        self.call(self.lib.tok_decode, tokens_json)
    }
}

impl Drop for TokHandle {
    fn drop(&mut self) {
        unsafe { (self.lib.tok_destroy)(self.ptr) };
    }
}
