from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appearance import Appearance

if TYPE_CHECKING:
  from state import State
  from uniformbuffer import UniformBlock

class Material (Appearance):


    def __init__ (self, **values: Any) -> None:
      self._values: dict[str, Any] = dict(values)
      self._revision: int = 0

    def write_fields (self, block: UniformBlock) -> None:

      for name, value in self._values.items():
        if block.has_field(name):
          block.set(name, value)

    def set (self, name: str, value: Any) -> None:

      self._values[name] = value
      self._mark_dirty()

    def get (self, name: str, default: Any = None) -> Any:

      return self._values.get(name, default)

    def get_values (self) -> dict[str, Any]:

      return self._values.copy()

    def _mark_dirty (self) -> None:

      self._revision += 1

    def get_revision (self) -> int:
      return self._revision

    def load (self, st: State) -> None:

      st.get_shader().bind_material(st, self)

    def unload (self, st: State) -> None:

      st.get_shader().unbind_material(st)
