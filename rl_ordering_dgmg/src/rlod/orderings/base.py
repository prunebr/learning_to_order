from __future__ import annotations
from typing import Protocol, List
from rlod.graphs.types import GraphSample


class OrderingStrategy(Protocol):
    def order(self, sample: GraphSample) -> List[int]:
        ...
