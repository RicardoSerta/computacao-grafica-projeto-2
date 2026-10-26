from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appearance import Appearance

if TYPE_CHECKING:
  from state import State

class TextureSet (Appearance):


  def __init__ (self, items: list[Any]) -> None:

    self.items = items

  def load (self, st: State) -> None:

    st.get_shader().bind_texture_set(st, self)

  def unload (self, st: State) -> None:

    st.get_shader().unbind_texture_set(st)
