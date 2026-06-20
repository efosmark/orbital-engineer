from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from orbitalengineer.ui import ui_config

@runtime_checkable
class Tone(Protocol):
    
    min = 1.0
    max = 1.0
    
    def generate(self, buffer: NDArray, sample_rate:int, sample_index):...
    
    def get_phase(self, frequency:float, sample_rate, sample_index):
        sample_size = 2.0 * np.pi * frequency / sample_rate
        return ((np.arange(ui_config.BUFFER_SAMPLES) + sample_index) * sample_size) % (2.0 * np.pi)

    def __mul__(self, other:Tone|float|int) -> 'Mixer':
        return Mixer(self, other)
    
    def __add__(self, other:Tone|float|int) -> 'Sum':
        return Sum(self, other)
        

class Sine(Tone):
    
    def __init__(self, frequency:float):
        self.frequency = frequency
        
    def generate(self, buffer: NDArray, sample_rate:int, sample_index):
        phase = self.get_phase(self.frequency, sample_rate, sample_index)
        np.sin(phase, out=buffer)


class PWM(Tone):
    
    def __init__(self, frequency:float, duty_cycle:float):
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        
    def generate(self, buffer: NDArray, sample_rate:int, sample_index):
        phase = self.get_phase(self.frequency, sample_rate, sample_index)
        buffer.fill(-1)
        buffer[phase <= (2.0 * np.pi * self.duty_cycle)] = 1.0

class Square(PWM):
    def __init__(self, frequency:float):
        super().__init__(frequency, 0.5)


class CompositeTone(Tone):...


class Sum(CompositeTone):
    
    def __init__(self, *tones:Tone|int|float):
        self.tones = [
            t if isinstance(t, Tone) else Sine(t)
            for t in tones
        ]
        
    def generate(self, buffer: NDArray, sample_rate:int, sample_index):
        for tone in self.tones:
            b1 = np.zeros(ui_config.BUFFER_SAMPLES, dtype=np.float64)
            tone.generate(b1, sample_rate, sample_index)
            b1 /= len(self.tones)
            np.add(buffer, b1, out=buffer)

class Mixer(CompositeTone):
    
    def __init__(self, fundamental:Tone, secondary:Tone|float|int):
        self.fundamental = fundamental
        self.secondary = secondary if isinstance(secondary, Tone) else Sine(secondary)
    
    def generate(self, buffer: NDArray, sample_rate:int, sample_index):
        
        b1 = np.zeros(ui_config.BUFFER_SAMPLES, dtype=np.float64)
        self.fundamental.generate(b1, sample_rate, sample_index)
        
        b2 = np.zeros(ui_config.BUFFER_SAMPLES, dtype=np.float64)
        self.secondary.generate(b2, sample_rate, sample_index)
        
        np.multiply(b1, b2, out=buffer)
        #self.rescale(buffer)

class OnOff(CompositeTone):
    def __init__(self, fundamental:Tone, pwm_tone:Tone):
        self.fundamental = fundamental
        self.pwm_tone = pwm_tone

    def generate(self, buffer: NDArray, sample_rate:int, sample_index):
        self.fundamental.generate(buffer, sample_rate, sample_index)
        b2 = np.zeros(ui_config.BUFFER_SAMPLES, dtype=np.float64)
        self.pwm_tone.generate(b2, sample_rate, sample_index)
        
        #sub_one = np.argwhere(b2 < 1)
        #if len(sub_one) > 0:
        #    first = sub_one[0]
            
        
        
        buffer[b2 < 1] = 0.0