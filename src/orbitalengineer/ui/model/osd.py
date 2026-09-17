from dataclasses import dataclass
import time
from typing import Any
from orbitalengineer.ui.gtk4 import GObject


@dataclass
class OSDMessage:
    message:str
    duration:float
    category:str|None = None
    start:float|None = None
    description:str|None = None
    fade_in:bool = True
    now:float = 0


class OnScreenDisplayModel(GObject.GObject):
    props:Any
    osd_message = GObject.Property(type=object)
    
    def __init__(self):
        super().__init__()
        self.props.osd_message = []
        self._current_message = None     

    def add_message(self, message:str, duration:float=1, category:str|None=None, desc:str|None=None):
        # Remove any messages of the same category (e.g. allow overwriting same type)
        fade_in = len([m for m in self.osd_message if m.start is not None]) == 0
        self.osd_message = [m for m in self.osd_message if m.category != category]
        self.osd_message.append(OSDMessage(message, duration, category, description=desc, fade_in=fade_in))

    def get_message(self) -> OSDMessage|None:
        self._current_message = None
        if len(self.osd_message) == 0:
            return
        
        now = time.monotonic()
        for m in [*self.osd_message]:
            if m.start is None:
                m.start = now
            if now - m.start >= m.duration:
                self.osd_message.remove(m)
            if m.start > now:
                continue
            self._current_message = m
            break

        if self._current_message is not None:
            self._current_message.now = now
        return self._current_message