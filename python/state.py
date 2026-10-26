from __future__ import annotations

from typing import TYPE_CHECKING, Any

import graphics_math as gm
import wgpu

if TYPE_CHECKING:
  from camera import Camera
  from shader import Shader
  from pipeline import Pipeline

class State:


  def __init__ (self, camera: Camera, device: wgpu.GPUDevice, render_pass: wgpu.GPURenderPassEncoder,
                canvas_size: tuple[int, int]) -> None:

    self.camera = camera
    self.device = device
    self.render_pass = render_pass
    self.canvas_size = canvas_size
    self.pipeline: list[Pipeline] = []
    self._active_pipeline: Pipeline | None = None
    self.bind_group_stacks: dict[int, list[wgpu.GPUBindGroup]] = {}


    self._matrix_stack: list[gm.Mat4] = []
    self._matrix_version = 0


    self.derived: dict[str, gm.Mat4] = {}

    self._view: gm.Mat4 = camera.get_view_matrix()
    self._proj: gm.Mat4 = camera.get_proj_matrix(canvas_size)
    self._view_proj: gm.Mat4 = self._proj @ self._view
    self._matrix_shaders: list[Any] = []
    self._inverse_view: gm.Mat4 | None = None

  def push_pipeline (self, pip: Pipeline) -> None:

    self.pipeline.append(pip)

  def pop_pipeline (self) -> None:

    self.pipeline.pop()

  def get_pipeline (self) -> Pipeline:

    if not self.pipeline:
      raise RuntimeError("Pipeline not defined")
    return self.pipeline[-1]

  def get_shader (self) -> Shader:

    return self.get_pipeline().get_shader()

  def get_active_pipeline (self) -> Pipeline | None:

    return self._active_pipeline

  def set_active_pipeline (self, pip: Pipeline) -> None:

    self._active_pipeline = pip

  def push_bind_group (self, group_index: int, bind_group: wgpu.GPUBindGroup) -> None:

    stack = self.bind_group_stacks.setdefault(group_index, [])
    stack.append(bind_group)
    self.render_pass.set_bind_group(group_index, bind_group)

  def pop_bind_group (self, group_index: int) -> None:

    stack = self.bind_group_stacks[group_index]
    stack.pop()
    if stack:
      self.render_pass.set_bind_group(group_index, stack[-1])

  def register_matrix_shader (self, shader: Any) -> None:

    if shader not in self._matrix_shaders:
      self._matrix_shaders.append(shader)

  def get_matrix_shaders (self) -> list[Any]:

    return self._matrix_shaders

  def push_matrix (self, mat: gm.Mat4) -> None:

    self._matrix_stack.append(self.get_matrix() @ mat)
    self._matrix_version += 1

  def push_matrix_absolute (self, mat: gm.Mat4) -> None:

    self._matrix_stack.append(mat)
    self._matrix_version += 1

  def pop_matrix (self) -> None:

    self._matrix_stack.pop()
    self._matrix_version += 1

  def get_matrix (self) -> gm.Mat4:

    return self._matrix_stack[-1] if self._matrix_stack else gm.mat4(1.0)

  def get_matrix_version (self) -> int:

    return self._matrix_version

  def get_camera (self) -> Camera:
    return self.camera

  def get_view_matrix (self) -> gm.Mat4:

    return self._view

  def get_proj_matrix (self) -> gm.Mat4:

    return self._proj

  def get_view_proj_matrix (self) -> gm.Mat4:

    return self._view_proj

  def get_inverse_view_matrix (self) -> gm.Mat4:

    if self._inverse_view is None:
      self._inverse_view = gm.inverse(self._view)
    return self._inverse_view

  def load_matrices (self) -> None:

    shd = self.get_shader()
    mat = self.get_matrix()
    needs_vertex = shd.declares_matrix_field("vertex")
    needs_normal = shd.declares_matrix_field("normal")
    if needs_vertex or needs_normal:
      mv = mat
      if shd.get_lighting_space() == "camera":
        mv = self.get_view_matrix() @ mv
      if needs_vertex:
        self.derived["vertex"] = mv
      if needs_normal:


        try:
          self.derived["normal"] = gm.transpose(gm.inverse(mv))
        except Exception:
          self.derived["normal"] = gm.mat4(1.0)

  def unload_matrices (self) -> None:

    self.derived.clear()

  def get_derived (self, name: str) -> gm.Mat4 | None:

    return self.derived.get(name)
