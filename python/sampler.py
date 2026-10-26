from __future__ import annotations

from typing import Any

import wgpu

class Sampler:


  def __init__ (self, device: wgpu.GPUDevice, varname: str,
                address_mode_u: str = "repeat", address_mode_v: str = "repeat",
                mag_filter: str = "linear", min_filter: str = "linear",
                mipmap_filter: str = "linear", compare: str | None = None) -> None:

    self.varname = varname
    kwargs: dict[str, Any] = dict(
      address_mode_u=address_mode_u, address_mode_v=address_mode_v,
      mag_filter=mag_filter, min_filter=min_filter, mipmap_filter=mipmap_filter,
    )
    if compare is not None:
      kwargs["compare"] = compare
    self.sampler: wgpu.GPUSampler = device.create_sampler(**kwargs)

  @property
  def resource (self) -> wgpu.GPUSampler:

    return self.sampler
