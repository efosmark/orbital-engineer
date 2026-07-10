
from typing import Any, Protocol, Self

class SupportsToDict(Protocol):
    def to_dict(self) -> dict[str, Any]: ...

class SupportsFromDict(Protocol):
    @classmethod
    def from_dict(cls, input:dict) -> Self:...