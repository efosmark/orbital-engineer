

class OrbitalEngineException(Exception):...


class InitKernelException(OrbitalEngineException):
    def __init__(self):
        super().__init__("Cannot initialize kernels until CL device is selected.")