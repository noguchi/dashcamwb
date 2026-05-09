import numpy as np

Matrix3x3 = np.ndarray  # type alias for clarity

def identity() -> Matrix3x3:
    return np.eye(3)

def from_diag(r: float, g: float, b: float) -> Matrix3x3:
    return np.diag([r, g, b]).astype(np.float64)

def _validate(M: Matrix3x3) -> None:
    if M.shape != (3, 3):
        raise ValueError(f"expected 3x3 matrix, got shape {M.shape}")

def compose(left: Matrix3x3, right: Matrix3x3) -> Matrix3x3:
    _validate(left)
    _validate(right)
    return left @ right
