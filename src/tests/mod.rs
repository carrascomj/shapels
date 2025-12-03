//! Each of the submodules in the tests module points to a python
//! file that is parsed and analysed for testing that the expected
//! hovers and diagnostics are produced.

use lsp_types::Position;
use shapels::Analysis;

mod test_aggr;
mod test_multifile;
mod test_multiply;
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

pub fn assert_hover_expected(src: &str, analysis: &Analysis, pat: &str, expected: &str) {
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
        .expect(format!("Hover info failed for pat {pat} with expected shape {expected}").as_str());
    let shape = hover.shape.as_ref().unwrap();
    assert_eq!(shape.dim_string(), expected);
}
