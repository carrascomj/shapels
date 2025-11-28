use crate::Shape;

/// Matrix multiplication shape inference shared by `@` and `torch.mm`.
/// Keeps all leading dims of left except the last, then appends all trailing dims of right except the first.
pub fn infer_matmul(left: &Shape, right: &Shape) -> Result<Shape, String> {
    if left.dims.is_empty() || right.dims.is_empty() {
        return Err("Matmul requires both operands to have shapes".into());
    }
    let left_inner = left.dims.last().unwrap();
    let right_inner = right.dims.first().unwrap();
    if left_inner != right_inner {
        return Err(format!(
            "Matmul inner dimensions mismatch: {} vs {}",
            left_inner, right_inner
        ));
    }
    let mut dims: Vec<String> = left.dims[..left.dims.len() - 1].to_vec();
    dims.extend_from_slice(&right.dims[1..]);
    Ok(Shape {
        dtype: left.dtype.clone().or(right.dtype.clone()),
        dims,
    })
}
