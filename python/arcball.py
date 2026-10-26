from __future__ import annotations

import math
from typing import Any

import graphics_math as gm

class Arcball:


  def __init__ (self, distance: float) -> None:

    self.distance = distance
    self.x0: float = 0
    self.y0: float = 0
    self.mat: gm.Mat4 = gm.mat4(1)
    self._dragging: bool = False

  def attach (self, canvas: Any) -> None:

    def on_down (event: Any) -> None:

      self._dragging = True


      w, h = canvas.get_logical_size()
      self.init_mouse_motion(event["x"], h - event["y"])
    def on_up (event: Any) -> None:

      self._dragging = False
    def on_move (event: Any) -> None:

      if self._dragging:
        w, h = canvas.get_logical_size()
        self.accumulate_mouse_motion(event["x"], h - event["y"], w, h)
    canvas.add_event_handler(on_down, "pointer_down")
    canvas.add_event_handler(on_up, "pointer_up")
    canvas.add_event_handler(on_move, "pointer_move")

  def init_mouse_motion (self, x0: float, y0: float) -> None:

    self.x0 = x0
    self.y0 = y0

  def accumulate_mouse_motion (self, x: float, y: float, width: float, height: float) -> None:

    if x==self.x0 and y==self.y0:
      return
    ux, uy, uz = Map(width,height,self.x0,self.y0)
    vx, vy, vz = Map(width,height,x,y)
    self.x0 = x
    self.y0 = y
    ax = uy*vz - uz*vy
    ay = uz*vx - ux*vz
    az = ux*vy - uy*vx
    cross_len = math.sqrt(ax*ax+ay*ay+az*az)
    if cross_len < 1e-9:
      return


    dot = ux*vx + uy*vy + uz*vz
    theta = 2 * math.atan2(cross_len, dot)

    m = gm.mat4(1)
    m = gm.translate(m,gm.vec3(0,0,-self.distance))
    m = gm.rotate(m,theta,gm.vec3(ax,ay,az))
    m = gm.translate(m,gm.vec3(0,0,self.distance))
    self.mat = m @ self.mat

  def get_matrix (self) -> gm.Mat4:

    return self.mat

  def translate (self, dx: float, dy: float, dz: float) -> None:

    m = gm.mat4(1)
    m = gm.translate(m,gm.vec3(dx*self.distance,dy*self.distance,dz*self.distance))
    self.mat = m @ self.mat

def Map (width: float, height: float, x: float, y: float) -> tuple[float, float, float]:

  if width < height:
    r = width/2
  else:
    r = height/2
  X = (x - width/2) / r
  Y = (y - height/2) / r
  l = math.sqrt(X*X + Y*Y)
  if l <= 1:
    Z = math.sqrt(1 - l*l)
  else:
    X /= l
    Y /= l
    Z = 0
  return (X,Y,Z)
