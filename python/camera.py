from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State

class Camera:


  def get_proj_matrix (self, canvas_size: tuple[int, int]) -> gm.Mat4:

    return gm.mat4(1)

  def get_view_matrix (self) -> gm.Mat4:

    return gm.mat4(1)

  def load (self, st: State) -> None:

    pass

  def unload (self, st: State) -> None:

    pass
