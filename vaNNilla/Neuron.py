from .random import Random


class Neuron:
    def __init__(self):
        self.rng = Random()
        self.weight = 0
