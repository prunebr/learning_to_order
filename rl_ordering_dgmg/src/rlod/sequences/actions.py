from __future__ import annotations

from enum import IntEnum


class ActionType(IntEnum):
    """
    Ações no estilo DGMG (teacher forcing).

    Sequência típica:
      (ADD_NODE, -1)
      repetir k vezes:
        (ADD_EDGE, -1)
        (CHOOSE_DEST, dest_index)
      (STOP_EDGE, -1)
    ...
    no fim:
      (STOP_NODE, -1)
    """
    ADD_NODE = 0
    ADD_EDGE = 1
    CHOOSE_DEST = 2
    STOP_EDGE = 3
    STOP_NODE = 4
