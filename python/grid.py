from __future__ import annotations

import numpy as np

class Grid:


  def __init__ (self, nx: int, ny: int) -> None:

    if nx <= 0 or ny <= 0:
      raise ValueError(f"Grid needs nx > 0 and ny > 0, got nx={nx}, ny={ny}")
    self.nx: int = nx
    self.ny: int = ny

    self.coords: np.ndarray = np.empty(2*self.vertex_count(), dtype = 'float32')
    self.texcoords: np.ndarray = np.empty(2*self.vertex_count(), dtype = 'float32')
    dx = 1 / nx
    dy = 1 / ny
    nc = 0
    for j in range(0,ny+1):
      for i in range(0,nx+1):
        self.coords[nc+0] = i*dx
        self.coords[nc+1] = j*dy
        self.texcoords[nc+0] = i*dx
        self.texcoords[nc+1] = 1 - j*dy
        nc += 2


    def findex (i: int, j: int, nx: int) -> int:

      return j*(nx+1) + i

    self.indices: np.ndarray = np.empty(self.index_count(), dtype = 'uint32')
    ni = 0
    for j in range(0,ny):
      for i in range(0,nx):
        self.indices[ni+0] = findex(i,j,nx)
        self.indices[ni+1] = findex(i+1,j,nx)
        self.indices[ni+2] = findex(i+1,j+1,nx)
        self.indices[ni+3] = findex(i,j,nx)
        self.indices[ni+4] = findex(i+1,j+1,nx)
        self.indices[ni+5] = findex(i,j+1,nx)
        ni += 6

  def get_nx (self) -> int:
    return self.nx

  def get_ny (self) -> int:
    return self.ny

  def vertex_count (self) -> int:

    return (self.nx+1)*(self.ny+1)

  def get_coords (self) -> np.ndarray:

    return self.coords

  def get_texcoords (self) -> np.ndarray:

    return self.texcoords

  def index_count (self) -> int:

    return 6*self.nx*self.ny

  def get_indices (self) -> np.ndarray:

    return self.indices
