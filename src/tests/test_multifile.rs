//! Tests multifile support using .venv files and relative paths at test_data/multi*.py

use shapels::{ModuleCache, analyze_file, analyze_source_at_path_with_cache};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tests::{extract_test_case, is_hover_dtype, is_hover_expected};
use shapels::analyze_source_at_path;

const PY_MULTI_CALLER: &str = include_str!("../../test_data/multi_caller.py");
const PY_REPRO: &str = include_str!("../../test_data/import_callee.py");

#[test]
fn multifile_hover_follows_imported_function() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 1);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B X O"));
}

#[test]
fn multi_file_wrong_shape_arg_is_reported_on_caller() {
    let src = extract_test_case(PY_MULTI_CALLER, 9);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
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
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z =", "B A X"));
}

#[test]
fn multi_file_reads_callee_float_type_hint() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 4);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_dtype(&src, &analysis, "z =", "F"));
}

#[test]
fn multi_file_reads_callee_int_type_hint() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 5);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_dtype(&src, &analysis, "z =", "Int"));
}

#[test]
fn multi_file_runs_inference_on_callee_with_tuple_return() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 6);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z, ", "B A X"));
}

#[test]
fn multi_file_runs_inference_on_callee_with_complex_tuple_return() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 7);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "z, ", "B A X"));
}

#[test]
fn multi_file_runs_inference_on_torch_module_instances() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 8);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
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
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    assert!(analysis.diagnostics.is_empty());
}

#[test]
fn multi_file_runs_inference_on_torch_module_method() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_MULTI_CALLER, 10);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    // no diagnostics expected
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "out =", "B T O K"));
}

#[test]
fn multi_file_annotated_ellipsis_function_preserves_prefix() {
    let src = extract_test_case(PY_MULTI_CALLER, 11);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(&src, &analysis, "y =", "2 3 OutDim"));
}

#[test]
fn multi_file_annotated_ellipsis_function_tuple_destructuring() {
    let src = extract_test_case(PY_MULTI_CALLER, 12);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y, residual =",
        "2 3 5 OutDim"
    ));
    assert!(is_hover_expected(&src, &analysis, "residual =", "2 3 5 7"));
}

#[test]
fn annotated_ellipsis_with_two_dims_also_works() {
    let src = extract_test_case(PY_MULTI_CALLER, 13);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    assert_eq!(analysis.diagnostics.len(), 0);
    assert!(is_hover_expected(
        &src,
        &analysis,
        "y,",
        "2 3 OutHeight OutWidth"
    ));
    assert!(is_hover_expected(
        &src,
        &analysis,
        "residual =",
        "2 3 OutHeight OutWidth"
    ));
}

#[test]
fn annotated_ellipsis_with_insufficient_dims_emits_diagnostic() {
    let src = extract_test_case(PY_MULTI_CALLER, 14);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/multi_caller.py"));
    assert_eq!(analysis.diagnostics.len(), 1);
}

#[test]
fn imported_function_does_not_show_diagnostic_on_callee() {
    // analyze entire caller file from disk so relative module resolution works
    let src = extract_test_case(PY_REPRO, 1);
    let analysis = analyze_source_at_path(&src, Path::new("test_data/import_repro.py"));
    // no diagnostics expected
    println!("{:?}", analysis.diagnostics);
    assert_eq!(analysis.diagnostics.len(), 0);
}

#[test]
fn persistent_module_cache_reloads_changed_imported_self_attr_refs() {
    let temp_dir = temp_test_dir("cache-invalidation");
    let caller_path = temp_dir.join("caller.py");
    let callee_path = temp_dir.join("imported.py");

    fs::write(&callee_path, imported_module_source("First")).expect("write initial callee");
    fs::write(&caller_path, caller_module_source()).expect("write caller");

    let caller_src = fs::read_to_string(&caller_path).expect("read caller");
    let mut cache = ModuleCache::new();

    let analysis = analyze_source_at_path_with_cache(&caller_src, &caller_path, &mut cache);
    assert!(is_hover_expected(&caller_src, &analysis, "z =", "B X O"));

    cache.update_file_source(&callee_path, imported_module_source("Second"));

    let updated = analyze_source_at_path_with_cache(&caller_src, &caller_path, &mut cache);
    assert!(is_hover_expected(&caller_src, &updated, "z =", "B X P"));

    let _ = fs::remove_dir_all(&temp_dir);
}

fn temp_test_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("shapels-{prefix}-{nanos}"));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn caller_module_source() -> String {
    r#"
import torch
from imported import Outer


def run():
    B, X, R = 2, 3, 5
    x = torch.zeros(B, X, R)
    model = Outer()
    z = model(x)
"#
    .trim_start()
    .to_string()
}

fn imported_module_source(proj_class: &str) -> String {
    format!(
        r#"
import torch
from jaxtyping import Float as F
from torch import Tensor as T


class First(torch.nn.Module):
    def forward(self, x: F[T, "B X R"]) -> F[T, "B X O"]:
        return x


class Second(torch.nn.Module):
    def forward(self, x: F[T, "B X R"]) -> F[T, "B X P"]:
        return x


class Outer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = {proj_class}()

    def forward(self, x: F[T, "B X R"]):
        return self.proj(x)
"#
    )
}
