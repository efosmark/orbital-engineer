import enum

class CollisionStrategy(enum.Enum):
    NONE = enum.auto()
    MERGE = enum.auto()
    BOUNCE = enum.auto()
