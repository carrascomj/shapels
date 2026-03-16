use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::path::Path;

use shapels::{ModuleCache, analyze_source_at_path, analyze_source_at_path_with_cache};

const PY_MULTIFILE_DATA: &str = include_str!("../test_data/multi_caller.py");
const PY_VIEW_DATA: &str = include_str!("../test_data/view.py");
const PY_UNET_DATA: &str = include_str!("../test_data/unet.py");

pub fn extract_test_case(py_data: &'static str, n: usize) -> String {
    let marker = format!("# test {n}");
    let mut buf = Vec::new();
    let mut capture = false;
    for line in py_data.lines() {
        if line.trim() == marker {
            capture = true;
            continue;
        }
        if capture && line.starts_with("# test ") {
            break;
        }
        if capture {
            buf.push(line);
        }
    }
    buf.join("\n")
}

fn analyse_and_count_diagnostics(src: &str, path: &Path) -> usize {
    let analysis = analyze_source_at_path(&src, path);
    analysis.diagnostics.len()
}

fn analyse_cached_and_count_diagnostics(
    src: &str,
    path: &Path,
    module_cache: &mut ModuleCache,
) -> usize {
    let analysis = analyze_source_at_path_with_cache(&src, path, module_cache);
    analysis.diagnostics.len()
}

fn parse_unet(c: &mut Criterion) {
    let path = Path::new("test_data/unet.py");
    let mut module_cache = ModuleCache::new();
    c.bench_function("Unet analysis", |b| {
        b.iter(|| analyse_cached_and_count_diagnostics(PY_UNET_DATA, &path, &mut module_cache))
    });
}

fn multifile_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultiFile");
    let path = Path::new("test_data/multi_caller.py");
    for test_idx in 1..11 {
        group.bench_with_input(
            BenchmarkId::from_parameter(test_idx),
            &test_idx,
            |b, &test_idx| {
                let test_src = extract_test_case(PY_MULTIFILE_DATA, test_idx);
                b.iter(|| analyse_and_count_diagnostics(test_src.as_str(), &path));
            },
        );
    }
    group.finish();
}

fn view_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("View");
    let path = Path::new("test_data/view.py");
    for test_idx in 1..19 {
        group.bench_with_input(
            BenchmarkId::from_parameter(test_idx),
            &test_idx,
            |b, &test_idx| {
                let test_src = extract_test_case(PY_VIEW_DATA, test_idx);
                b.iter(|| analyse_and_count_diagnostics(test_src.as_str(), &path));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, parse_unet, multifile_tests, view_tests);
criterion_main!(benches);
