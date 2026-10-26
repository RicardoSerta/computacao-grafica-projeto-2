from __future__ import annotations

import math
from typing import TYPE_CHECKING

import graphics_math as gm
from camera import *

if TYPE_CHECKING:
  from arcball import Arcball
  from node import Node

class Camera3D (Camera):


  def __init__(self, x: float, y: float, z: float) -> None:

    self.ortho: bool = False
    self.fovy: float = 45
    self.znear: float = 0.1
    self.zfar: float = 1000
    self.center: gm.Vec3 = gm.vec3(0,0,0)
    self.eye: gm.Vec3 = gm.vec3(x,y,z)
    self.up: gm.Vec3 = gm.vec3(0,1,0)
    self.arcball: Arcball | None = None
    self.reference: Node | None = None

  def set_angle (self, fovy: float) -> None:

    self.fovy = fovy

  def get_angle (self) -> float:

    return self.fovy

  def set_z_planes (self, znear: float, zfar: float) -> None:

    self.znear = znear
    self.zfar = zfar

  def set_center (self, x: float, y: float, z: float) -> None:

    self.center = gm.vec3(x,y,z)

  def get_center (self) -> gm.Vec3:

    return self.center

  def set_eye (self, x: float, y: float, z: float) -> None:

    self.eye = gm.vec3(x,y,z)

  def get_eye (self) -> gm.Vec3:

    return self.eye

  def set_up_dir (self, x: float, y: float, z: float) -> None:

    self.up = gm.vec3(x,y,z)

  def set_ortho (self, flag: bool) -> None:

    self.ortho = flag

  def create_arcball (self) -> Arcball:

    from arcball import Arcball
    d = gm.length(self.eye-self.center)
    self.arcball = Arcball(d)
    return self.arcball

  def get_arcball (self) -> Arcball | None:

    return self.arcball

  def set_reference (self, ref: Node) -> None:

    self.reference = ref

  def get_proj_matrix (self, canvas_size: tuple[int, int]) -> gm.Mat4:


    w, h = canvas_size
    if w <= 0 or h <= 0:
      raise ValueError(f"get_proj_matrix needs a canvas_size with both dimensions > 0, got {canvas_size}")
    ratio = w/h
    if not self.ortho:
      return gm.perspective(gm.radians(self.fovy),ratio,self.znear,self.zfar)
    else:
      dist = gm.distance(self.eye,self.center)
      height = dist * math.tan(gm.radians(self.fovy)/2)
      width = height / h * w
      return gm.ortho(-width,width,-height,height,self.znear,self.zfar)

  def get_view_matrix (self) -> gm.Mat4:


    view = gm.mat4(1.0)
    if self.arcball:
      view = view @ self.arcball.get_matrix()
    view = view @ gm.look_at(self.eye,self.center,self.up)
    if (self.reference):
        view = view @ gm.inverse(self.reference.get_model_matrix())
    return view
