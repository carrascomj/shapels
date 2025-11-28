mod test_multiply;

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
