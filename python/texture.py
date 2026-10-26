from __future__ import annotations

from PIL import Image
import numpy as np
import graphics_math as gm
import wgpu

class Texture:


  def __init__ (self, device: wgpu.GPUDevice, varname: str, filename: str | None, texel: gm.FloatArray | None = None, width: int = 1, height: int = 1, format: str = "rgba8unorm-srgb") -> None:

    self.varname = varname
    if filename:


      img = Image.open(filename)
      img = img.convert("RGBA")
      data = np.array(img)
      width, height = img.size
    elif texel is None:
      data = np.zeros((height, width, 4), dtype='uint8')
    elif np.shape(texel) == (3,):
      width, height = 1, 1
      data = np.array([[[texel[0]*255, texel[1]*255, texel[2]*255, 255]]], dtype='uint8')
    elif np.shape(texel) == (4,):
      width, height = 1, 1
      data = np.array([[[texel[0]*255, texel[1]*255, texel[2]*255, texel[3]*255]]], dtype='uint8')
    else:
      raise RuntimeError("Invalid Texture parameters")
    self.width: int = width
    self.height: int = height


    self.tex: wgpu.GPUTexture = device.create_texture(
      size=(width, height, 1), format=format,
      usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
      {"texture": self.tex}, data.tobytes(),
      {"bytes_per_row": width * 4, "rows_per_image": height}, (width, height, 1),
    )
    self.view: wgpu.GPUTextureView = self.tex.create_view()

  def get_texture (self) -> wgpu.GPUTexture:
    return self.tex

  def get_width (self) -> int:
    return self.width

  def get_height (self) -> int:
    return self.height

  @property
  def resource (self) -> wgpu.GPUTextureView:

    return self.view
