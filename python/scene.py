from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from node import Node
  from engine import Engine
  from state import State

class Scene:


  def __init__ (self, root: Node) -> None:
    self.root = root
    self.engines: list[Engine] = []

  def get_root (self) -> Node:
    return self.root

  def add_engine (self, engine: Engine) -> None:
    self.engines.append(engine)

  def update (self, dt: float) -> None:

    for e in self.engines:
      e.update(dt)

  def render (self, state: State) -> None:

    self.root.render(state)
