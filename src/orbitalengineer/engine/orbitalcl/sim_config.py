from dataclasses import dataclass, fields
from typing import Self
from orbitalengineer.engine import logger, config

@dataclass
class SimConfig:
    COEF_OF_RESTITUTION:float = config.COEF_OF_RESTITUTION
    DEFAULT_DT_BASE:float = config.DEFAULT_DT_BASE
    GRAV_CONSTANT:float = config.GRAV_CONSTANT
    EPS_DIST:float = config.EPS_DIST
    EPS_TIME:float = config.EPS_TIME
    DV_MAX:float = config.DV_MAX
    MAX_NUM_CONTACTS_PER_BODY:int = config.MAX_NUM_CONTACTS_PER_BODY
    ENABLE_PROFILING:bool = config.ENABLE_PROFILING
    TARGET_TICKS_PER_SECOND:int = config.TARGET_TICKS_PER_SECOND

    def as_build_contants(self):
        defs = dict()
        for field in fields(self):
            value = getattr(self, field.name, None)
            if value is None:
                continue
            if isinstance(value, float):
                value = f"{value:.06f}f"
            elif isinstance(value, bool):
                value = f"{int(value)}"
            else:
                value = f"{value}"
            defs[field.name] = value
        logger.info("Using build constants: %s", [ f'{k}={v}' for k,v in defs.items() ])
        return [ f'-D{k}={v}' for k,v in defs.items() ]

    @classmethod
    def from_dict(cls, d:dict) -> Self:
        return cls(**d)