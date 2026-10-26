from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from state import State
  from pipeline import Pipeline
  from transform import Transform
  from appearance import Appearance
  from shape import Shape

class Node:


  def __init__ (self, pipeline: Pipeline | None = None, trf: Transform | None = None, apps: list[Appearance] | None = None, shps: list[Shape] | None = None, nodes: list[Node] | None = None) -> None:

    self.parent: Node | None = None
    self.pipeline = pipeline
    self.trf = trf
    self.apps: list[Appearance] = apps or []
    self.shps: list[Shape] = shps or []
    self.nodes: list[Node] = []
    if nodes:
      for n in nodes:
        self.add_node(n)

  def set_pipeline (self, pipeline: Pipeline) -> None:

    self.pipeline = pipeline

  def get_pipeline (self) -> Pipeline | None:

    return self.pipeline

  def set_transform (self, trf: Transform | None) -> None:

    self.trf = trf

  def add_appearance (self, app: Appearance) -> None:

    self.apps.append(app)

  def add_shape (self, shp: Shape) -> None:

    self.shps.append(shp)

  def add_node (self, node: Node) -> None:

    if node.parent is not None:
      raise ValueError("Node already has a parent - remove it from its current parent first")
    ancestor: Node | None = self
    while ancestor is not None:
      if ancestor is node:
        raise ValueError("Node is self or an ancestor of self - would create a cycle")
      ancestor = ancestor.parent
    self.nodes.append(node)
    node.set_parent(self)

  def remove_node (self, node: Node) -> None:

    if node not in self.nodes:
      raise ValueError("node is not a child of this Node")
    self.nodes.remove(node)
    node.parent = None

  def set_parent (self, parent: Node) -> None:

    self.parent = parent

  def get_parent (self) -> Node | None:

    return self.parent

  def get_matrix (self) -> gm.Mat4:

    if self.trf:
      return self.trf.get_matrix()
    else:
      return gm.mat4(1.0)

  def get_model_matrix (self) -> gm.Mat4:

    mat = self.get_matrix()
    node = self.get_parent()
    while node:
      mat = node.get_matrix() @ mat
      node = node.get_parent()
    return mat

  def render (self, st: State) -> None:


    if self.pipeline:
      self.pipeline.load(st)
    if self.trf:
      self.trf.load(st)
    for app in self.apps:
      app.load(st)

    if len(self.shps) > 0:


      st.load_matrices()
      for shp in self.shps:
        shp.draw(st)
      st.unload_matrices()
    for node in self.nodes:
      node.render(st)

    for app in reversed(self.apps):
      app.unload(st)
    if self.trf:
      self.trf.unload(st)
    if self.pipeline:
      self.pipeline.unload(st)
