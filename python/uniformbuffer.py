from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import graphics_math as gm
import numpy as np
import wgpu

if TYPE_CHECKING:
  from state import State


_ALIGN_SIZE = {
  "i32": (4, 4),
  "f32": (4, 4),
  "vec2": (8, 8),
  "vec3": (16, 12),
  "vec4": (16, 16),
  "mat4x4": (16, 64),
}


WgslType = Any
Schema = list[tuple[str, WgslType]]

_COMPONENT_COUNT = {
  "vec2": 2,
  "vec3": 3,
  "vec4": 4,
  "mat4x4": 16,
}

_ZERO = {
  "i32": 0, "f32": 0.0, "vec2": gm.vec2(0), "vec3": gm.vec3(0),
  "vec4": gm.vec4(0), "mat4x4": gm.mat4(0),
}

def _align (offset: int, alignment: int) -> int:
  return (offset + alignment - 1) // alignment * alignment

def _pack (value: Any, wgsl_type: str) -> bytes:
  if wgsl_type == "i32":
    return struct.pack("<i", int(value))
  if wgsl_type == "f32":
    return struct.pack("<f", float(value))
  if wgsl_type == "mat4x4":
    return gm.mat4_bytes(value)
  if wgsl_type in _COMPONENT_COUNT:
    n = _COMPONENT_COUNT[wgsl_type]
    vector = np.asarray(value, dtype="<f4")
    if vector.shape != (n,):
      raise ValueError(f"{wgsl_type} value must have shape ({n},), got {vector.shape}")
    return vector.tobytes()
  raise TypeError("Unsupported wgsl_type in uniformbuffer: " + wgsl_type)

def _type_align_size (wgsl_type: WgslType) -> tuple[int, int]:

  if isinstance(wgsl_type, list):
    _, _, natural_size, align = _layout_schema(wgsl_type)
    return align, _align(natural_size, align)
  return _ALIGN_SIZE[wgsl_type]

def _layout_schema (schema: Schema) -> tuple[dict[str, int], dict[str, str], int, int]:

  offsets: dict[str, int] = {}
  types: dict[str, str] = {}
  offset = 0
  align = 1
  for name, wgsl_type in schema:
    field_align, size = _type_align_size(wgsl_type)
    align = max(align, field_align)
    offset = _align(offset, field_align)
    if isinstance(wgsl_type, list):
      sub_offsets, sub_types, _, _ = _layout_schema(wgsl_type)
      for sub_name, sub_offset in sub_offsets.items():
        offsets[f"{name}.{sub_name}"] = offset + sub_offset
        types[f"{name}.{sub_name}"] = sub_types[sub_name]
    else:
      offsets[name] = offset
      types[name] = wgsl_type
    offset += size
  return offsets, types, offset, align

def _compute_layout (schema: Schema) -> tuple[dict[str, int], dict[str, str], int]:

  offsets, types, natural_size, _ = _layout_schema(schema)
  return offsets, types, _align(natural_size, 16)

def _pack_row (values: dict[str, Any], types: dict[str, str], offsets: dict[str, int], stride: int) -> bytes:

  row = bytearray(stride)
  for name, wgsl_type in types.items():
    data = _pack(values.get(name, _ZERO[wgsl_type]), wgsl_type)
    off = offsets[name]
    row[off:off + len(data)] = data
  return bytes(row)


class StorageArray:


  def __init__ (self, device: wgpu.GPUDevice, schema: Schema,
                layout: wgpu.GPUBindGroupLayout, group_index: int, max_rows: int, binding: int = 0) -> None:
    self.device = device
    self.schema = schema
    self.group_index = group_index
    self.max_rows = max_rows
    self.offsets, self._types, self.stride = _compute_layout(schema)
    self.count = 0
    self._staging = bytearray(self.stride * max_rows)
    self.buffer = device.create_buffer(
      size=self.stride * max_rows, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    self.bind_group = device.create_bind_group(
      layout=layout,
      entries=[{"binding": binding, "resource": {"buffer": self.buffer, "offset": 0, "size": self.stride * max_rows}}],
    )

  def append (self, values: dict[str, Any]) -> int:

    if self.count >= self.max_rows:
      raise RuntimeError(
        f"StorageArray for group {self.group_index} exceeded its {self.max_rows}-row capacity "
        "(see Shader's max_instances) - raise it or draw fewer distinct instances per frame"
      )
    row = self.count
    off = row * self.stride
    self._staging[off:off + self.stride] = _pack_row(values, self._types, self.offsets, self.stride)
    self.count += 1
    return row

  def flush (self) -> None:

    if self.count == 0:
      return
    self.device.queue.write_buffer(self.buffer, 0, bytes(self._staging[:self.count * self.stride]))

  def reset (self) -> None:

    self.count = 0


class UniformBlock:


  def __init__ (self, device: wgpu.GPUDevice, schema: Schema,
                layout: wgpu.GPUBindGroupLayout, group_index: int, binding: int = 0) -> None:
    self.device = device
    self.schema = schema
    self.group_index = group_index
    self.offsets, self._types, self.size = _compute_layout(schema)
    self._pending: bytearray | None = None
    self.buffer = device.create_buffer(
      size=self.size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )
    self.bind_group = device.create_bind_group(
      layout=layout,
      entries=[{"binding": binding, "resource": {"buffer": self.buffer, "offset": 0, "size": self.size}}],
    )

  def has_field (self, name: str) -> bool:

    return name in self.offsets

  def begin (self) -> None:

    self._pending = bytearray(self.size)

  def set (self, name: str, value: Any) -> None:

    if name not in self.offsets:
      raise ValueError(f"'{name}' is not a field of this shader's group")
    data = _pack(value, self._types[name])
    off = self.offsets[name]
    assert self._pending is not None
    self._pending[off:off + len(data)] = data

  def end (self) -> None:

    assert self._pending is not None
    self.device.queue.write_buffer(self.buffer, 0, bytes(self._pending))
    self._pending = None
