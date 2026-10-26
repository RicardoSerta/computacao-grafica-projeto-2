from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import graphics_math as gm

if TYPE_CHECKING:
  from state import State
  from node import Node
  from uniformbuffer import UniformBlock


class Light(ABC):


    def __init__ (self, space: str = "world", **values: Any) -> None:
      if space not in ("world", "camera", "local"):
        raise ValueError("Light space must be 'world', 'camera', or 'local'")
      self.space = space
      self.reference: Node | None = None
      self._values: dict[str, Any] = {
        "light_ambient": gm.vec3(0.2,0.2,0.2),
        "light_diffuse": gm.vec3(0.8,0.8,0.8),
        "light_specular": gm.vec3(1.0,1.0,1.0),
      }
      self._values.update(values)

    def set (self, name: str, value: Any) -> None:

      self._values[name] = value

    def get (self, name: str, default: Any = None) -> Any:
      return self._values.get(name, default)

    def get_values (self) -> dict[str, Any]:

      return self._values.copy()

    def set_ambient (self, r: float, g: float, b: float) -> None:
      self.set("light_ambient", gm.vec3(r,g,b))

    def set_diffuse (self, r: float, g: float, b: float) -> None:
      self.set("light_diffuse", gm.vec3(r,g,b))

    def set_specular (self, r: float, g: float, b: float) -> None:
      self.set("light_specular", gm.vec3(r,g,b))

    def set_reference (self, reference: Node | None) -> None:

      if self.space != "local":
        raise ValueError("A reference node can only be assigned to a local-space light")
      self.reference = reference

    def get_reference (self) -> Node | None:
      return self.reference

    @abstractmethod
    def map (self, matrix: gm.Mat4) -> dict[str, Any]:

      raise NotImplementedError()

    def get_mapping_matrix (self, st: State, lighting_space: str) -> gm.Mat4:

      if lighting_space not in ("world", "camera"):
        raise ValueError("Shader lighting space must be 'world' or 'camera'")

      if self.space == "local":
        if self.reference is None:
          raise RuntimeError("Local-space light has no reference node")
        model = self.reference.get_model_matrix()
        if lighting_space == "world":
          return model
        return st.get_view_matrix() @ model

      if self.space == lighting_space:
        return gm.mat4(1.0)
      if self.space == "world":
        return st.get_view_matrix()
      return st.get_inverse_view_matrix()

    def write_fields (self, block: UniformBlock, st: State, lighting_space: str) -> None:

      matrix = self.get_mapping_matrix(st, lighting_space)
      values = self.get_values()
      values.update(self.map(matrix))
      for name, value in values.items():
        if block.has_field(name):
          block.set(name, value)
