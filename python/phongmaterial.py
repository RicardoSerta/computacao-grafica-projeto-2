from __future__ import annotations

import graphics_math as gm
from material import Material

class PhongMaterial(Material):


    def __init__ (self, r: float, g: float, b: float, opacity: float = 1.0) -> None:
      Material.__init__(
        self,
        base_color=gm.vec3(r,g,b),
        opacity=opacity,
        specular_color=gm.vec3(1,1,1),
        shininess=32.0,
      )

    def set_base_color (self, r: float, g: float, b: float) -> None:
      self.set("base_color", gm.vec3(r,g,b))

    def set_specular (self, r: float, g: float, b: float) -> None:
      self.set("specular_color", gm.vec3(r,g,b))

    def set_shininess (self, shi: float) -> None:
      self.set("shininess", shi)

    def set_opacity (self, opacity: float) -> None:

      self.set("opacity", opacity)
