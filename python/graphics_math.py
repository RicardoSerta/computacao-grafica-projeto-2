from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
Vec2 = FloatArray
Vec3 = FloatArray
Vec4 = FloatArray
Mat4 = FloatArray
VectorLike = Sequence[float] | FloatArray


def vec2(x: float = 0.0, y: float | None = None) -> Vec2:

  if y is None:
    y = x
  return np.array((x, y), dtype=np.float32)


def vec3(x: float = 0.0, y: float | None = None, z: float | None = None) -> Vec3:

  if y is None and z is None:
    y = z = x
  elif y is None or z is None:
    raise TypeError("vec3 needs either one value or all of x, y, and z")
  return np.array((x, y, z), dtype=np.float32)


def vec4(
  x: float = 0.0,
  y: float | None = None,
  z: float | None = None,
  w: float | None = None,
) -> Vec4:

  if y is None and z is None and w is None:
    y = z = w = x
  elif y is None or z is None or w is None:
    raise TypeError("vec4 needs either one value or all of x, y, z, and w")
  return np.array((x, y, z, w), dtype=np.float32)


def mat4(diagonal: float = 1.0) -> Mat4:

  return np.eye(4, dtype=np.float32) * np.float32(diagonal)


def mat4_from_columns(c0: VectorLike, c1: VectorLike, c2: VectorLike, c3: VectorLike) -> Mat4:

  return np.column_stack(
    (_vector(c0, 4, "c0"), _vector(c1, 4, "c1"),
     _vector(c2, 4, "c2"), _vector(c3, 4, "c3"))
  ).astype(np.float32, copy=False)


def radians(degrees: float) -> float:

  return math.radians(degrees)


def dot(a: VectorLike, b: VectorLike) -> float:
  return float(np.dot(a, b))


def cross(a: VectorLike, b: VectorLike) -> Vec3:
  return np.asarray(np.cross(a, b), dtype=np.float32)


def length(v: VectorLike) -> float:
  return float(np.linalg.norm(v))


def distance(a: VectorLike, b: VectorLike) -> float:
  return length(np.asarray(a) - np.asarray(b))


def normalize(v: VectorLike) -> FloatArray:

  value = np.asarray(v, dtype=np.float32)
  magnitude = length(value)
  if magnitude == 0.0:
    raise ValueError("cannot normalize a zero-length vector")
  return np.asarray(value / magnitude, dtype=np.float32)


def transpose(matrix: Mat4) -> Mat4:
  return np.asarray(_mat4(matrix, "matrix").T, dtype=np.float32)


def inverse(matrix: Mat4) -> Mat4:
  return np.asarray(np.linalg.inv(_mat4(matrix, "matrix")), dtype=np.float32)


def translation(offset: VectorLike) -> Mat4:

  x, y, z = _components(offset, 3, "offset")
  result = mat4()
  result[:3, 3] = (x, y, z)
  return result


def scaling(factors: VectorLike) -> Mat4:

  x, y, z = _components(factors, 3, "factors")
  result = mat4()
  result[0, 0] = x
  result[1, 1] = y
  result[2, 2] = z
  return result


def rotation(angle: float, axis: VectorLike) -> Mat4:

  x, y, z = normalize(_vector(axis, 3, "axis"))
  c = math.cos(angle)
  s = math.sin(angle)
  t = 1.0 - c
  return np.array(
    (
      (t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0.0),
      (t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0.0),
      (t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0.0),
      (0.0,         0.0,         0.0,           1.0),
    ),
    dtype=np.float32,
  )


def translate(matrix: Mat4, offset: VectorLike) -> Mat4:

  return _product(matrix, translation(offset))


def scale(matrix: Mat4, factors: VectorLike) -> Mat4:

  return _product(matrix, scaling(factors))


def rotate(matrix: Mat4, angle: float, axis: VectorLike) -> Mat4:

  return _product(matrix, rotation(angle, axis))


def look_at(eye: VectorLike, center: VectorLike, up: VectorLike) -> Mat4:

  eye_v = _vector(eye, 3, "eye")
  center_v = _vector(center, 3, "center")
  forward = normalize(center_v - eye_v)
  side = normalize(np.cross(forward, _vector(up, 3, "up")))
  corrected_up = np.cross(side, forward)

  result = mat4()
  result[0, :3] = side
  result[1, :3] = corrected_up
  result[2, :3] = -forward
  result[0, 3] = -np.dot(side, eye_v)
  result[1, 3] = -np.dot(corrected_up, eye_v)
  result[2, 3] = np.dot(forward, eye_v)
  return result


def perspective(fovy: float, aspect: float, near: float, far: float) -> Mat4:

  if not 0.0 < fovy < math.pi:
    raise ValueError(f"fovy must be between 0 and pi radians, got {fovy}")
  if aspect <= 0.0:
    raise ValueError(f"aspect must be > 0, got {aspect}")
  _validate_perspective_planes(near, far)

  focal_length = 1.0 / math.tan(fovy / 2.0)
  result = np.zeros((4, 4), dtype=np.float32)
  result[0, 0] = focal_length / aspect
  result[1, 1] = focal_length
  result[2, 2] = far / (near - far)
  result[2, 3] = far * near / (near - far)
  result[3, 2] = -1.0
  return result


def ortho(
  left: float,
  right: float,
  bottom: float,
  top: float,
  near: float = -1,
  far: float = 1,
) -> Mat4:

  if left == right:
    raise ValueError("left and right must differ")
  if bottom == top:
    raise ValueError("bottom and top must differ")
  if near == far:
    raise ValueError("near and far must differ")

  result = mat4()
  result[0, 0] = 2.0 / (right - left)
  result[1, 1] = 2.0 / (top - bottom)
  result[2, 2] = 1.0 / (near - far)
  result[0, 3] = -(right + left) / (right - left)
  result[1, 3] = -(top + bottom) / (top - bottom)
  result[2, 3] = near / (near - far)
  return result


def mat4_bytes(matrix: Mat4) -> bytes:

  value = _mat4(matrix, "matrix").astype("<f4", copy=False)
  return value.tobytes(order="F")


def _vector(value: VectorLike, size: int, name: str) -> FloatArray:
  result = np.asarray(value, dtype=np.float32)
  if result.shape != (size,):
    raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
  return result


def _components(value: VectorLike, size: int, name: str) -> tuple[float, ...]:
  return tuple(float(component) for component in _vector(value, size, name))


def _mat4(value: Mat4, name: str) -> Mat4:
  result = np.asarray(value, dtype=np.float32)
  if result.shape != (4, 4):
    raise ValueError(f"{name} must have shape (4, 4), got {result.shape}")
  return result


def _product(left: Mat4, right: Mat4) -> Mat4:
  return np.asarray(_mat4(left, "left") @ _mat4(right, "right"), dtype=np.float32)


def _validate_perspective_planes(near: float, far: float) -> None:
  if near <= 0.0:
    raise ValueError(f"near must be > 0, got {near}")
  if far <= near:
    raise ValueError(f"far must be greater than near, got near={near}, far={far}")
