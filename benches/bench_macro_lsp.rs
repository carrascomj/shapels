use criterion::{Criterion, criterion_group, criterion_main};
use shapels::{ModuleCache, analyze_file_with_cache, analyze_source_at_path_with_cache};
use std::hint::black_box;
use std::path::Path;

const MODEL_SRC: &str = include_str!("../test_data/bench_lsp/model.py");
const MODEL_ALT_SRC: &str = include_str!("../test_data/bench_lsp/model_alt.py");
const BLOCKS_SRC: &str = include_str!("../test_data/bench_lsp/blocks.py");
const BLOCKS_ALT_SRC: &str = include_str!("../test_data/bench_lsp/blocks_alt.py");

fn warm_cached_workspace(c: &mut Criterion) {
    let path = Path::new("test_data/bench_lsp/model.py");
    let mut cache = ModuleCache::new();

    let _ = analyze_file_with_cache(path, &mut cache);

    c.bench_function("MacroWarmCache", |b| {
        b.iter(|| black_box(analyze_file_with_cache(path, &mut cache).diagnostics.len()))
    });
}

fn imported_module_invalidation(c: &mut Criterion) {
    let caller_path = Path::new("test_data/bench_lsp/model.py");
    let blocks_path = Path::new("test_data/bench_lsp/blocks.py");
    let mut cache = ModuleCache::new();
    let mut use_alt = false;

    let _ = analyze_source_at_path_with_cache(MODEL_SRC, caller_path, &mut cache);

    c.bench_function("MacroImportInvalidation", |b| {
        b.iter(|| {
            use_alt = !use_alt;
            let next = if use_alt { BLOCKS_ALT_SRC } else { BLOCKS_SRC };
            cache.update_file_source(blocks_path, next.to_string());
            black_box(
                analyze_source_at_path_with_cache(MODEL_SRC, caller_path, &mut cache)
                    .diagnostics
                    .len(),
            )
        })
    });
}

fn caller_edit_warm_cache(c: &mut Criterion) {
    let caller_path = Path::new("test_data/bench_lsp/model.py");
    let mut cache = ModuleCache::new();
    let mut use_alt = false;

    let _ = analyze_source_at_path_with_cache(MODEL_SRC, caller_path, &mut cache);

    c.bench_function("MacroCallerEditWarm", |b| {
        b.iter(|| {
            use_alt = !use_alt;
            let next = if use_alt { MODEL_ALT_SRC } else { MODEL_SRC };
            cache.update_file_source(caller_path, next.to_string());
            black_box(
                analyze_source_at_path_with_cache(next, caller_path, &mut cache)
                    .diagnostics
                    .len(),
            )
        })
    });
}

criterion_group!(
    macro_lsp_benches,
    warm_cached_workspace,
    imported_module_invalidation,
    caller_edit_warm_cache
);
criterion_main!(macro_lsp_benches);
