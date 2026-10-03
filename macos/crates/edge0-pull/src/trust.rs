//! Compiled-in public-key anchors; hosting sites are not trusted.

use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};

use crate::canonical::canonical_bytes;
use crate::error::PullError;
use edge0_core::manifest::Manifest;

pub use base64::engine::general_purpose::STANDARD as B64;

#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub key_id: &'static str,
    pub pubkey_b64: &'static str,
}

pub const ANCHORS: &[Anchor] = &[
    Anchor {
        key_id: "+y4LSPkaSzk=", // edge0-registry-2026
        pubkey_b64: "VIq4ssoyjcpM+PcOFN6k42YZASdocY3tFYxilcUYW0o=",
    },
    Anchor {
        key_id: "pvsAPXUAzqE=", // dev-test-2026
        pubkey_b64: "ACNtc1fOxqUBHUWOpTp0jBQT9+FB8eglCuEeFOwu0DM=",
    },
];

pub fn verify(manifest: &Manifest) -> Result<&'static str, PullError> {
    let value = serde_json::to_value(manifest)
        .map_err(|e| PullError::new("E-MANIFEST-SIG", format!("manifest reserialize failed: {e}")))?;
    let msg = canonical_bytes(&value);
    let sig_bytes = B64
        .decode(&manifest.signature)
        .map_err(|_| PullError::new("E-MANIFEST-SIG", "signature is not valid base64 (64 bytes)"))?;
    if sig_bytes.len() != 64 {
        return Err(PullError::new("E-MANIFEST-SIG", "signature length is not 64 bytes"));
    }
    let sig = Signature::from_bytes(sig_bytes.as_slice().try_into().unwrap());
    let candidates: Vec<&Anchor> = match manifest.signer_key_id.as_deref() {
        Some(kid) => ANCHORS.iter().filter(|a| a.key_id == kid).collect(),
        None => ANCHORS.iter().collect(),
    };
    let n_cand = candidates.len();
    for a in candidates {
        let Ok(raw) = B64.decode(a.pubkey_b64) else {
            continue;
        };
        let Ok(bytes): Result<[u8; 32], _> = raw.try_into() else {
            continue;
        };
        if let Ok(vk) = VerifyingKey::from_bytes(&bytes) {
            if vk.verify_strict(&msg, &sig).is_ok() {
                return Ok(a.key_id);
            }
        }
    }
    Err(PullError::new(
        "E-MANIFEST-SIG",
        format!("signature check failed: untrusted source or tampered content ({n_cand} candidate anchors)"),
    ))
}

pub fn sign_value(manifest: &mut serde_json::Value, secret: &[u8; 32], key_id: &str) {
    use ed25519_dalek::{Signer, SigningKey};
    if let Some(obj) = manifest.as_object_mut() {
        obj.remove("signature");
        obj.remove("signer_key_id");
    }
    let msg = canonical_bytes(manifest);
    let sk = SigningKey::from_bytes(secret);
    let sig = sk.sign(&msg);
    if let Some(obj) = manifest.as_object_mut() {
        obj.insert(
            "signature".into(),
            serde_json::json!(B64.encode(sig.to_bytes())),
        );
        obj.insert("signer_key_id".into(), serde_json::json!(key_id));
    }
}
