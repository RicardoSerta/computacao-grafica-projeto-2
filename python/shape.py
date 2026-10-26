from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from state import State

class Shape:


  def draw (self, st: State) -> None:

    raise NotImplementedError()
