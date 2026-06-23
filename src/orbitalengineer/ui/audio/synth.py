import struct

import numpy as np
from orbitalengineer.ui.audio import tone
from orbitalengineer.ui import ui_config

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib #  type:ignore

Gst.init(None)


SAMPLE_MIN = -30000
SAMPLE_MAX = 30000

class ToneSynthController:
    
    def __init__(self):
        self.sample_index = 0
        self.tones:list[tone.Tone] = []
        self.pipeline = Gst.parse_launch(ui_config.AUDIO_PIPELINE)

        self.appsrc = self.pipeline.get_by_name(ui_config.AUDIO_PIPELINE_NAME) # type:ignore
        self.appsrc.set_property("caps", Gst.Caps.from_string(ui_config.GST_CAPS))
        self.appsrc.connect("need-data", self.need_data)
        self.pipeline.set_state(Gst.State.PLAYING)

    def need_data(self, src, length):
        buffer = np.zeros(ui_config.BUFFER_SAMPLES, dtype=np.float64)
        
        tones = tone.Sum(*self.tones)
        tones.generate(buffer, ui_config.SAMPLE_RATE, self.sample_index)
        
        # Scale the data for int16
        data = np.clip(np.int16(np.multiply(buffer, SAMPLE_MAX)), SAMPLE_MIN, SAMPLE_MAX)
        
        print(data)    
        # Pack the sample data
        data = struct.pack("<" + "h" * len(data), *data) # type:ignore
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        if buf is None: raise Exception("No buffer")
        buf.fill(0, data)
        buf.pts = Gst.util_uint64_scale(self.sample_index, Gst.SECOND, ui_config.SAMPLE_RATE)
        buf.duration = Gst.util_uint64_scale(ui_config.BUFFER_SAMPLES, Gst.SECOND, ui_config.SAMPLE_RATE)

        self.sample_index += ui_config.BUFFER_SAMPLES
        src.emit("push-buffer", buf)


if __name__ == "__main__":
    from orbitalengineer.ui.audio import tone 
    ctl = ToneSynthController()
    
    
    ctl.tones.append(tone.OnOff(((tone.Sine(440)) * 3), tone.PWM(0.5, 0.66)))
    
    #ctl.tones.append(dial_tone)
    # ctl.tones.append(tone.Sum(
    #     tone.Sine(440),
    #     #tone.Sine(880, SAMPLE_RATE),
    #     tone.Sine(440*2),
    #     tone.Sine(440*3),
    #     tone.Sine(440*4)
    # ))
    
    
    #ctl.tones.append(dial_tone)
    
    # ctl.tones.append(
    #     OnOffTone(
    #         dial_tone,
    #         PWMTone(0.4, 0.66, SAMPLE_RATE)
    #     )
    # )
    
    #ctl.tones.append(PureSineWave(640, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(650, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(660, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(670, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(580, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(580, SAMPLE_RATE))
    #ctl.tones.append(PureSineWave(990, SAMPLE_RATE))

    ctl.pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    loop.run()