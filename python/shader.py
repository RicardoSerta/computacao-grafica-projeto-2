from __future__ import annotations

from typing import TYPE_CHECKING, Any

import graphics_math as gm
import wgpu

import shaderutl as sutl
import wgslreflect
from uniformbuffer import StorageArray, UniformBlock

if TYPE_CHECKING:
  from state import State
  from light import Light
  from material import Material
  from textureset import TextureSet

class Shader:


  def __init__ (self, device: wgpu.GPUDevice, wgsl_path: str,
                light: Light | None = None, space: str = "camera", max_instances: int = 1024) -> None:

    self.device = device
    self.light = light
    self.space = space

    code = sutl.readfile(wgsl_path)
    self.module: wgpu.GPUShaderModule = device.create_shader_module(code=code)
    self._reflection = wgslreflect.ShaderReflection(wgsl_path)

    self._layouts: dict[int, wgpu.GPUBindGroupLayout] = {}
    for g in self._reflection.group_indices():
      self._layouts[g] = device.create_bind_group_layout(entries=self._reflection.layout_entries(g))

    self._matrix_array: StorageArray | None = None
    try:
      matrix_group = self._reflection.uniform_var_group("matrix")
    except KeyError:
      matrix_group = None
    if matrix_group is not None:
      fields = self._reflection.storage_array_fields(matrix_group)
      self._matrix_array = StorageArray(device, fields, self._layouts[matrix_group], matrix_group, max_instances)
    self._frame_state: State | None = None
    self._matrix_tick = -1
    self._matrix_row = -1


    try:
      self._material_group: int | None = self._reflection.uniform_var_group("material")
    except KeyError:
      self._material_group = None
    self._materials: dict[Material, UniformBlock] = {}
    self._material_revisions: dict[Material, int] = {}


    try:
      global_group = self._reflection.uniform_var_group("global")
    except KeyError:
      global_group = None
    self._global_block: UniformBlock | None = None
    if global_group is not None:
      fields = self._reflection.uniform_fields(global_group)
      self._global_block = UniformBlock(device, fields, self._layouts[global_group], global_group)
    self._values: dict[str, Any] = {}


    texture_groups = [
      g for g in self._reflection.group_indices()
      if self._reflection.uniform_fields(g) is None and self._reflection.storage_array_fields(g) is None
    ]
    if len(texture_groups) > 1:
      raise RuntimeError(
        f"shader declares more than one texture/sampler-only group {texture_groups} - "
        "merge them into a single @group"
      )
    self._texture_group: int | None = texture_groups[0] if texture_groups else None
    self._texture_sets: dict[TextureSet, wgpu.GPUBindGroup] = {}


    self._vertex_buffers: list[dict] = []


  def commit_matrix (self, st: State) -> int:

    if self._matrix_array is None:
      raise RuntimeError("commit_matrix called on a shader with no \"matrix\" storage array declared")
    if st is not self._frame_state:
      self._frame_state = st
      self._matrix_array.reset()
      self._matrix_tick = -1
      self._matrix_row = -1
    st.register_matrix_shader(self)
    latest = st.get_matrix_version()
    if latest != self._matrix_tick:
      values = {}
      for name, _ in self._matrix_array.schema:
        v = st.get_derived(name)
        if v is not None:
          values[name] = v
      self._matrix_row = self._matrix_array.append(values)
      self._matrix_tick = latest
    return self._matrix_row

  def flush_matrices (self) -> None:

    if self._matrix_array is not None:
      self._matrix_array.flush()

  def get_matrix_bind_group (self) -> wgpu.GPUBindGroup:

    if self._matrix_array is None:
      raise RuntimeError("get_matrix_bind_group called on a shader with no \"matrix\" storage array declared")
    return self._matrix_array.bind_group

  def get_matrix_group_index (self) -> int:

    if self._matrix_array is None:
      raise RuntimeError("get_matrix_group_index called on a shader with no \"matrix\" storage array declared")
    return self._matrix_array.group_index

  def declares_matrix_field (self, name: str) -> bool:

    if self._matrix_array is None:
      return False
    return any(n == name for n, _ in self._matrix_array.schema)


  def add_material (self, mat: Material) -> None:

    if self._material_group is None:
      raise RuntimeError("this shader has no \"material\" group")
    fields = self._reflection.uniform_fields(self._material_group)
    block = UniformBlock(self.device, fields, self._layouts[self._material_group], self._material_group)
    block.begin()
    mat.write_fields(block)
    block.end()
    self._materials[mat] = block
    self._material_revisions[mat] = mat.get_revision()

  def bind_material (self, st: State, mat: Material) -> None:

    if self._material_group is None:
      return
    block = self._materials.get(mat)
    if block is None:
      raise RuntimeError("Material was used under a shader it was never add_material()'d to")
    if self._material_revisions.get(mat) != mat.get_revision():
      block.begin()
      mat.write_fields(block)
      block.end()
      self._material_revisions[mat] = mat.get_revision()
    st.push_bind_group(block.group_index, block.bind_group)

  def unbind_material (self, st: State) -> None:

    if self._material_group is None:
      return
    st.pop_bind_group(self._material_group)


  def set_value (self, name: str, value: Any) -> None:

    self._values[name] = value

  def get_value (self, name: str, default: Any = None) -> Any:
    return self._values.get(name, default)

  def _compute_projection_matrix (self, st: State) -> gm.Mat4:

    if self.get_lighting_space() == "camera":
      return st.get_proj_matrix()
    return st.get_view_proj_matrix()

  def _compute_camera_position (self, st: State) -> gm.Vec4:

    camera_position = gm.vec4(0, 0, 0, 1)
    if self.get_lighting_space() == "world":
      mat = st.get_inverse_view_matrix()
      camera_position = mat @ camera_position
    return camera_position

  def commit_global (self, st: State) -> None:

    block = self._global_block
    if block is None:
      return
    block.begin()
    if block.has_field("projection"):
      block.set("projection", self._compute_projection_matrix(st))
    if block.has_field("camera_position"):
      block.set("camera_position", self._compute_camera_position(st))
    if self.light is not None:
      self.light.write_fields(block, st, self.get_lighting_space())
    for name, value in self._values.items():
      block.set(name, value)
    block.end()
    st.push_bind_group(block.group_index, block.bind_group)

  def unbind_global (self, st: State) -> None:

    if self._global_block is None:
      return
    st.pop_bind_group(self._global_block.group_index)


  def _resolve_binding (self, varname: str) -> tuple[int, int]:

    try:
      return self._reflection.texture_binding(varname)
    except KeyError:
      pass
    return self._reflection.sampler_binding(varname)

  def add_texture_set (self, ts: TextureSet) -> None:

    if self._texture_group is None:
      raise RuntimeError("this shader has no texture/sampler group")
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in ts.items:
      try:
        group, binding = self._resolve_binding(item.varname)
      except KeyError:
        raise ValueError(f"'{item.varname}' is not declared by this shader") from None
      if group != self._texture_group:
        raise ValueError(f"'{item.varname}' does not belong to this shader's texture group")
      if binding in seen:
        raise ValueError(f"binding {binding} supplied more than once")
      seen.add(binding)
      entries.append({"binding": binding, "resource": item.resource})
    expected = len(self._reflection.layout_entries(self._texture_group))
    if len(entries) != expected:
      raise ValueError(f"texture set covers {len(entries)} binding(s), shader declares {expected}")
    self._texture_sets[ts] = self.device.create_bind_group(
      layout=self._layouts[self._texture_group], entries=entries,
    )
    for item in ts.items:
      retain = getattr(item, "retain", None)
      if retain is not None:
        retain()

  def bind_texture_set (self, st: State, ts: TextureSet) -> None:

    if self._texture_group is None:
      return
    bind_group = self._texture_sets.get(ts)
    if bind_group is None:
      raise RuntimeError("TextureSet was used under a shader it was never add_texture_set()'d to")
    st.push_bind_group(self._texture_group, bind_group)

  def unbind_texture_set (self, st: State) -> None:

    if self._texture_group is None:
      return
    st.pop_bind_group(self._texture_group)


  def get_module (self) -> wgpu.GPUShaderModule:

    return self.module

  def get_bind_group_layouts (self) -> list[wgpu.GPUBindGroupLayout]:

    return [self._layouts[g] for g in self._reflection.group_indices()]

  def get_lighting_space (self) -> str:

    return self.space


  def set_vertex_buffers (self, buffers: list[dict]) -> None:

    resolved: list[dict] = []
    for buf in buffers:
      attributes = []
      for attr in buf["attributes"]:
        attr = dict(attr)
        if "var_name" in attr:
          name = attr.pop("var_name")
          try:
            attr["shader_location"] = self._reflection.vertex_location(name)
          except KeyError:
            raise ValueError(f"set_vertex_buffers references '{name}', but this shader "
                              f"doesn't declare a vertex input with that name") from None
        attributes.append(attr)
      resolved.append({**buf, "attributes": attributes})
    self._vertex_buffers = resolved

  def get_vertex_buffer_layout (self) -> list[dict]:

    covered: dict[int, None] = {}
    for buf in self._vertex_buffers:
      for attr in buf["attributes"]:
        location = attr["shader_location"]
        if location in covered:
          raise ValueError(f"Vertex buffer location {location} registered more than once via set_vertex_buffers")
        covered[location] = None
    missing = self._reflection.vertex_locations() - covered.keys()
    if missing:
      raise ValueError(f"Shader declares vertex input location(s) {sorted(missing)} "
                        f"with no matching set_vertex_buffers attribute")
    return self._vertex_buffers
