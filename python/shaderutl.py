from __future__ import annotations

def readfile (filename: str) -> str:

  with open(filename) as f:
    lines = f.readlines()
  text = ""
  for l in lines:
    text = text + l
  return text
