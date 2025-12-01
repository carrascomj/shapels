//! Each of the submodules in the tests module points to a python
//! file that is parsed and analysed for testing that the expected
//! hovers and diagnostics are produced.

mod test_aggr;
mod test_multifile;
mod test_multiply;
mod test_noop;
mod test_permute;
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
