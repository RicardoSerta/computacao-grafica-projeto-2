from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from state import State

class Appearance:


  def load (self, st: State) -> None:

    raise NotImplementedError()

  def unload (self, st: State) -> None:

    pass
