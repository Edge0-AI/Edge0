//! Download phases: `probe → select-source → preflight → fetch → verify → stage → commit → register`.

pub const PHASES: [&str; 8] = [
    "probe",
    "select-source",
    "preflight",
    "fetch",
    "verify",
    "stage",
    "commit",
    "register",
];

pub fn next(phase: &str) -> Option<&'static str> {
    let i = PHASES.iter().position(|p| *p == phase)?;
    PHASES.get(i + 1).copied()
}

pub fn is_valid(phase: &str) -> bool {
    PHASES.contains(&phase)
}
