from __future__ import annotations

from typing import TYPE_CHECKING, Any

import wgpu

if TYPE_CHECKING:
  from state import State
  from shader import Shader


_UNSET: Any = object()

_DEFAULT_PRIMITIVE = {"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"}
_DEFAULT_MULTISAMPLE = {"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False}
_DEFAULT_DEPTH_STENCIL = {
  "format": "depth24plus", "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less,
  "depth_bias": 0, "depth_bias_slope_scale": 0, "depth_bias_clamp": 0,
}

class Pipeline:


  def __init__ (self, shader: Shader, target_format: str | None,
                primitive: dict | None = None,
                depth_stencil: dict | None = _UNSET,
                multisample: dict | None = None,
                blend: dict | None = None,
                write_mask: int = wgpu.ColorWrite.ALL,
                stencil_reference: int | None = None) -> None:

    self.shader = shader
    self.device = shader.device
    self.target_format = target_format
    self.stencil_reference = stencil_reference

    self.primitive = {**_DEFAULT_PRIMITIVE, **(primitive or {})}
    self.multisample = {**_DEFAULT_MULTISAMPLE, **(multisample or {})}
    if depth_stencil is _UNSET:
      self.depth_stencil = dict(_DEFAULT_DEPTH_STENCIL)
    elif depth_stencil is None:
      self.depth_stencil = None
    else:
      self.depth_stencil = {**_DEFAULT_DEPTH_STENCIL, **depth_stencil}


    if target_format is None and self.depth_stencil is None:
      raise ValueError("Pipeline needs target_format or depth_stencil (or both) - neither given")
    fragment = None
    if target_format is not None:
      fragment = {"module": shader.get_module(), "entry_point": "fs_main",
                  "targets": [{"format": target_format, "blend": blend, "write_mask": write_mask}]}

    vertex_buffers = shader.get_vertex_buffer_layout()
    pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=shader.get_bind_group_layouts())

    self._gpu_pipeline = self.device.create_render_pipeline(
      layout=pipeline_layout,
      vertex={"module": shader.get_module(), "entry_point": "vs_main", "buffers": vertex_buffers},
      primitive=self.primitive,
      depth_stencil=self.depth_stencil,
      multisample=self.multisample,
      fragment=fragment,
    )

  def get_shader (self) -> Shader:
    return self.shader

  def activate (self, st: State) -> None:

    if st.get_active_pipeline() is self:
      return
    st.render_pass.set_pipeline(self._gpu_pipeline)
    st.render_pass.set_bind_group(self.shader.get_matrix_group_index(), self.shader.get_matrix_bind_group())
    st.set_active_pipeline(self)

  def load (self, st: State) -> None:

    st.push_pipeline(self)
    self.activate(st)
    self.shader.commit_global(st)
    st.render_pass.set_stencil_reference(self.stencil_reference if self.stencil_reference is not None else 0)

  def unload (self, st: State) -> None:

    st.pop_pipeline()
    top = st.pipeline[-1] if st.pipeline else None
    if top is not None:
      top.activate(st)
    self.shader.unbind_global(st)
    if top is not None:
      st.render_pass.set_stencil_reference(top.stencil_reference if top.stencil_reference is not None else 0)
