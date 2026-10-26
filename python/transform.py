from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State

class Transform:


  def __init__ (self) -> None:

    self.mat: gm.Mat4 = gm.mat4(1.0)

  def load_identity (self) -> None:

    self.mat = gm.mat4(1.0)

  def mult_matrix (self, mat: gm.Mat4) -> None:

    self.mat = self.mat @ mat

  def translate (self, x: float, y: float, z: float) -> None:

    self.mat = gm.translate(self.mat,gm.vec3(x,y,z))

  def scale (self, x: float, y: float, z: float) -> None:

    self.mat = gm.scale(self.mat,gm.vec3(x,y,z))

  def rotate (self, angle: float, x: float, y: float, z: float) -> None:

    self.mat = gm.rotate(self.mat,gm.radians(angle),gm.vec3(x,y,z))

  def get_matrix (self) -> gm.Mat4:
    return self.mat

  def load (self, st: State) -> None:

    st.push_matrix(self.mat)

  def unload (self, st: State) -> None:

    st.pop_matrix()
