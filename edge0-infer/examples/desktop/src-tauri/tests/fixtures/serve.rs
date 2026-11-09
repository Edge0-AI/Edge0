use std::{env, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let model = args.windows(2).find(|a| a[0] == "--model").map(|a| a[1].as_str()).unwrap_or("");
    if args.get(1).map(String::as_str) == Some("inspect") {
        if model == "missing" { eprintln!("Required asset is missing"); std::process::exit(1); }
        println!("{{\"adapter\":\"edge0_fixture\",\"files\":[{{\"bytes\":12}},{{\"bytes\":34}}]}}"); return;
    }
    if model == "error" { eprintln!("Model loading failed: invalid asset"); std::process::exit(1); }
    if model != "pending" { eprintln!("Listening: http://127.0.0.1:41325 model=fixture"); }
    loop { thread::sleep(Duration::from_millis(50)); }
}
