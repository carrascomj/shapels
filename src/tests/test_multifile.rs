//! Tests multifile support using .venv files and relative paths at test_data/multi*.py

use shapels::analyze_file;
use std::path::Path;

use crate::tests::assert_hover_expected;

#[test]
fn multifile_hover_follows_imported_function() {
    // analyze entire caller file from disk so relative module resolution works
    let path = Path::new("test_data/multi_caller.py");
    let analysis = analyze_file(path);
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);

    // find "z =" and ensure hover shows inferred shape B X O
    let src = std::fs::read_to_string(path).expect("read caller");
    assert_hover_expected(&src, &analysis, "z =", "B X O");
}

#[test]
fn venv_site_packages_module_resolves() {
    // prepend fake venv bin to PATH
    let bin_path = Path::new("test_data/.venv_fake/bin")
        .canonicalize()
        .unwrap();
    let mut new_path = format!("{}:", bin_path.display());
    if let Ok(old) = std::env::var("PATH") {
        new_path.push_str(&old);
    }
    unsafe {
        std::env::set_var("PATH", new_path);
    }

    let path = Path::new("test_data/venv_caller.py");
    let analysis = analyze_file(path);
    assert_eq!(analysis.diagnostics.len(), 0);

    let src = std::fs::read_to_string(path).expect("read caller");
    assert_hover_expected(&src, &analysis, "z =", "B X O");
}
