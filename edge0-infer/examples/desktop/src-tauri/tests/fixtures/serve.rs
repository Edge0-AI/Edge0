use std::{env, fs, path::Path, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let model = args.windows(2).find(|a| a[0] == "--model").map(|a| a[1].as_str()).unwrap_or("");
    // The fixture recognizes the four release names, leaving native manifests to the native path.
    let model_json = fs::read_to_string(Path::new(model).join("model.json")).unwrap_or_default();
    let names = [("edge0-145b-q3k", "edge0_145b"), ("edge0-145b-fp4", "edge0_145b"),
                 ("edge0-120b-q3k", "edge0_120b"), ("edge0-120b-fp4", "edge0_120b")];
    let found = names.iter().find(|(name, _)| model_json.contains(&format!("\"{name}\"")));
    let delegated = !model_json.contains("\"architecture\"") && model_json.contains("\"model_name\"") && found.is_some();
    let architecture = found.map(|(_, id)| *id).unwrap_or("edge0_fixture");
    if args.get(1).map(String::as_str) == Some("inspect") {
        if model == "missing" { eprintln!("Required asset is missing"); std::process::exit(1); }
        if delegated {
            println!("{{\"adapter\":\"{architecture}\",\"delegated\":true,\"engine\":\"edge0-strata\",\"ok\":true,\"files\":[{{\"bytes\":52}},{{\"bytes\":7}}]}}");
            return;
        }
        println!("{{\"adapter\":\"edge0_fixture\",\"files\":[{{\"bytes\":12}},{{\"bytes\":34}}]}}"); return;
    }
    if delegated && args.iter().any(|a| a == "--device" || a == "--prefill-device") {
        eprintln!("invalid_argument: this model runs on a CUDA GPU through edge0-strata; --device must be cuda or cuda:<n>");
        std::process::exit(2);
    }
    if model == "error" { eprintln!("Model loading failed: invalid asset"); std::process::exit(1); }
    if model != "pending" { eprintln!("Listening: http://127.0.0.1:41325 model=fixture"); }
    loop { thread::sleep(Duration::from_millis(50)); }
}
