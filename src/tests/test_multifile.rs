//! Tests multifile support using .venv files and relative paths at test_data/multi*.py

use shapels::analyze_file;
use std::path::Path;

use crate::analyze_source_at_path;
use crate::tests::{extract_test_case, is_hover_dtype, is_hover_expected};

const PY_MULTI_CALLER: &str = include_str!("../../test_data/multi_caller.py");

#[test]
fn multifile_hover_follows_imported_function() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 1);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X O"));
}

#[test]
fn multi_file_wrong_shape_arg_is_reported_on_caller() {
    let src = extract_test_case(PY_MULTI_CALLER, 9);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    if let Some(diag) = analysis.diagnostics.first() {
        // TODO: check if it is 0 or 1 indexed
        assert!(diag.range.start.line == 7 || diag.range.start.line == 8);
    } else {
        panic!("Expected at least one diagnostic.")
    }
}

#[test]
fn multi_file_runs_inference_on_callee() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 3);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B A X"));
}

#[test]
fn multi_file_reads_callee_float_type_hint() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 4);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_dtype(&src, &analysis, "z =", "F"));
}

#[test]
fn multi_file_reads_callee_int_type_hint() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 5);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_dtype(&src, &analysis, "z =", "Int"));
}

#[test]
fn multi_file_runs_inference_on_callee_with_tuple_return() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 6);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z, ", "B A X"));
}

#[test]
fn multi_file_runs_inference_on_callee_with_complex_tuple_return() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 7);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z, ", "B A X"));
}

#[test]
fn multi_file_runs_inference_on_torch_module_instances() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 8);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "b =", "B A A"));
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
    assert!(is_hover_expected(&src, &analysis, "z =", "B X O"));
}

#[test]
fn multi_file_alpha_equivalence_shape_arg_shape_checks() {
    let src = extract_test_case(PY_MULTI_CALLER, 2);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    assert!(analysis.diagnostics.is_empty());
}

#[test]
fn multi_file_runs_inference_on_torch_module_method() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 10);
    let analysis = analyze_source_at_path(&src, &Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B T O K"));
}
