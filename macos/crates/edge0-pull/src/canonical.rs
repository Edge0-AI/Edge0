//! Canonical JSON bytes for the signed object (keys sorted, no signature fields).

use serde_json::{Map, Value};

pub fn canonical_bytes(manifest: &Value) -> Vec<u8> {
    let mut v = manifest.clone();
    if let Some(obj) = v.as_object_mut() {
        obj.remove("signature");
        obj.remove("signer_key_id");
    }
    let sorted = sort_recursive(v);
    serde_json::to_vec(&sorted).expect("canonical serialize cannot fail")
}

fn sort_recursive(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(String, Value)> = map
                .into_iter()
                .map(|(k, val)| (k, sort_recursive(val)))
                .collect();
            pairs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            let mut out = Map::new();
            for (k, val) in pairs {
                out.insert(k, val);
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(sort_recursive).collect()),
        other => other,
    }
}
