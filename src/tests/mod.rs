//! Each of the submodules in the tests module points to a python
//! file that is parsed and analysed for testing that the expected
//! hovers and diagnostics are produced.

use lsp_types::Position;
use shapels::Analysis;

mod test_aggr;
mod test_branch;
mod test_class;
mod test_conv;
mod test_dtype;
mod test_index;
mod test_multifile;
mod test_multiply;
mod test_native_modules;
mod test_noop;
mod test_permute;
mod test_shape_assign;
mod test_view;

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

pub fn is_hover_expected(src: &str, analysis: &Analysis, pat: &str, expected: &str) -> bool {
    let mut saw_hover = false;
    for (idx, line) in src.lines().enumerate() {
        let mut search_idx = 0usize;
        while let Some(pos) = line[search_idx..].find(pat) {
            let col_idx = (search_idx + pos) as u32;
            let hover = analysis.hover(Position {
                line: idx as u32,
                character: col_idx,
            });
            if let Some(info) = hover {
                saw_hover = true;
                if let Some(shape) = info.shape.as_ref() {
                    let found_shape = shape.dim_string();
                    if found_shape == expected {
                        return true;
                    } else {
                        eprintln!("Found hover [{found_shape}] vs. [{expected}] expected");
                        return false;
                    }
                }
            }
            search_idx += pos + pat.len();
        }
    }
    if !saw_hover {
        panic!("Hover info failed for pat {pat} with expected shape {expected}");
    }
    false
}

fn is_hover_dtype(src: &str, analysis: &Analysis, pat: &str, expected_dtype: &str) -> bool {
    let mut line_idx = 0;
    let mut col_idx = 0;
    for (idx, line) in src.lines().enumerate() {
        if let Some(pos) = line.find(pat) {
            line_idx = idx as u32;
            col_idx = pos as u32;
            break;
        }
    }
    let hover = analysis
        .hover(Position {
            line: line_idx,
            character: col_idx,
        })
        .unwrap_or_else(|| {
            panic!("Hover info failed for pat {pat} with expected dtype {expected_dtype}")
        });
    let dtype = hover
        .shape
        .as_ref()
        .unwrap()
        .dtype
        .as_ref()
        .expect("Hover does not contain a dtype.");
    dtype == expected_dtype
}
