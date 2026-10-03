//! Tokenizer C ABI: string ↔ token ids and chat-template render. Callers own the JSON envelope.
use std::cell::RefCell;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use serde_json::{Map, Value};
use tokenizers::Tokenizer;

pub const ETOK_OK: c_int = 0;
pub const ETOK_ERR_INVALID: c_int = -1;
pub const ETOK_ERR_BUFFER: c_int = -2;

pub struct TokFace {
    tok: Tokenizer,
    template: String,
}

impl TokFace {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let tpath = model_dir.join("tokenizer.json");
        let cpath = model_dir.join("chat_template.jinja");
        let gpath = model_dir.join("config.json");
        for p in [&tpath, &cpath, &gpath] {
            if !p.is_file() {
                return Err(format!(
                    "tokenizer requires {p:?} (read-only tokenizer files; weights are not loaded here)"
                ));
            }
        }
        let tok =
            Tokenizer::from_file(&tpath).map_err(|e| format!("failed to parse tokenizer.json: {e}"))?;
        let template = std::fs::read_to_string(&cpath).map_err(|e| e.to_string())?;
        let _ = gpath;
        Ok(Self { tok, template })
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().iter().map(|i| *i as i32).collect())
            .unwrap_or_default()
    }

    pub fn decode(&self, tokens: &[i32]) -> Result<String, String> {
        let ids: Vec<u32> = tokens.iter().map(|t| *t as u32).collect();
        self.tok
            .decode(&ids, false)
            .map_err(|e| format!("decode failed: {e}"))
    }

    pub fn render(&self, ctx: &Map<String, Value>) -> Result<String, String> {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: minijinja::Value| -> Result<(), minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg.to_string(),
                ))
            },
        );
        let tmpl = env
            .template_from_str(&self.template)
            .map_err(|e| format!("template compile failed: {e:#}"))?;
        let value = minijinja::Value::from_serialize(ctx);
        tmpl.render(value)
            .map_err(|e| format!("template render failed: {e:#}"))
    }
}

thread_local! {
    static LAST_ERR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_err(msg: impl ToString) {
    LAST_ERR.with(|e| e.replace(msg.to_string()));
}

fn clear_err() {
    LAST_ERR.with(|e| e.borrow_mut().clear());
}

fn as_face<'a>(t: *mut c_void) -> Option<&'a TokFace> {
    if t.is_null() {
        set_err("etok: null handle");
        return None;
    }
    Some(unsafe { &*(t as *const TokFace) })
}

fn write_str(s: &str, buf: *mut c_char, buf_len: usize, written: *mut usize) -> c_int {
    let need = s.len();
    if !written.is_null() {
        unsafe { *written = need };
    }
    if buf.is_null() {
        return ETOK_OK;
    }
    if buf_len < need {
        return ETOK_ERR_BUFFER;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, need);
    }
    ETOK_OK
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn etok_create(model_dir: *const c_char) -> *mut c_void {
    if model_dir.is_null() {
        set_err("etok_create: model_dir is empty");
        return std::ptr::null_mut();
    }
    let dir = std::ffi::CStr::from_ptr(model_dir);
    let dir = match dir.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_err("etok_create: model_dir is not UTF-8");
            return std::ptr::null_mut();
        }
    };
    match TokFace::open(Path::new(dir)) {
        Ok(face) => {
            clear_err();
            Box::into_raw(Box::new(face)) as *mut c_void
        }
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
#[no_mangle]
pub extern "C" fn etok_destroy(t: *mut c_void) {
    if !t.is_null() {
        drop(unsafe { Box::from_raw(t as *mut TokFace) });
    }
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn etok_last_error(
    buf: *mut c_char,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    let msg = LAST_ERR.with(|e| e.borrow().clone());
    write_str(&msg, buf, buf_len, written)
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn etok_tokenize(
    t: *mut c_void,
    text: *const c_char,
    ids_out: *mut i32,
    ids_cap: usize,
    written: *mut usize,
) -> c_int {
    let Some(face) = as_face(t) else {
        return ETOK_ERR_INVALID;
    };
    if text.is_null() {
        set_err("etok_tokenize: text is empty");
        return ETOK_ERR_INVALID;
    }
    let Ok(s) = std::ffi::CStr::from_ptr(text).to_str() else {
        set_err("etok_tokenize: text is not UTF-8");
        return ETOK_ERR_INVALID;
    };
    let ids = face.encode(s);
    let n = ids.len();
    if !written.is_null() {
        *written = n;
    }
    if ids_out.is_null() {
        return ETOK_OK;
    }
    if ids_cap < n {
        return ETOK_ERR_BUFFER;
    }
    std::ptr::copy_nonoverlapping(ids.as_ptr(), ids_out, n);
    ETOK_OK
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn etok_decode(
    t: *mut c_void,
    ids: *const i32,
    n_ids: usize,
    buf: *mut c_char,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    let Some(face) = as_face(t) else {
        return ETOK_ERR_INVALID;
    };
    if ids.is_null() && n_ids > 0 {
        set_err("etok_decode: ids is null but length is non-zero");
        return ETOK_ERR_INVALID;
    }
    let tokens = if n_ids == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(ids, n_ids).to_vec()
    };
    match face.decode(&tokens) {
        Ok(s) => {
            clear_err();
            write_str(&s, buf, buf_len, written)
        }
        Err(e) => {
            set_err(e);
            ETOK_ERR_INVALID
        }
    }
}

/// # Safety
#[no_mangle]
pub unsafe extern "C" fn etok_render(
    t: *mut c_void,
    ctx_json: *const c_char,
    buf: *mut c_char,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    let Some(face) = as_face(t) else {
        return ETOK_ERR_INVALID;
    };
    if ctx_json.is_null() {
        set_err("etok_render: ctx_json is empty");
        return ETOK_ERR_INVALID;
    }
    let Ok(s) = std::ffi::CStr::from_ptr(ctx_json).to_str() else {
        set_err("etok_render: ctx_json is not UTF-8");
        return ETOK_ERR_INVALID;
    };
    let ctx: Map<String, Value> = match serde_json::from_str(s) {
        Ok(m) => m,
        Err(e) => {
            set_err(format!("invalid ctx_json: {e}"));
            return ETOK_ERR_INVALID;
        }
    };
    match face.render(&ctx) {
        Ok(out) => {
            clear_err();
            write_str(&out, buf, buf_len, written)
        }
        Err(e) => {
            set_err(e);
            ETOK_ERR_INVALID
        }
    }
}
