class Random:
    def __init__(self,
                 seed=1618033988):
        self.original_seed = seed
        self.seed = seed
        self.generator_active = False
        self.kill_generator = False
        self.rng = self.rand_range()

    def reset_seed(self):
        self.seed = self.original_seed

    def set_seed(self, new_seed):
        if self.generator_active:
            self.stop()
        self.original_seed = new_seed
        self.seed = new_seed

    def stop(self):
        self.kill_generator = True
        while self.generator_active:
            next(self.rng)
            print("Killing generator")
        self.kill_generator = False

    def next(self,
             min_val=0,
             max_val=1):
        return min_val + (max_val - min_val) * next(self.rng)

    def rand_range(self):
        """
        :return: a generator yielding floats between min_val and max_val, generated by a standard
                 Linear Congruential Generator (LCG)

                 seed = (seed * multiplier + offset) % modulus
                 u = seed / modulus

                 out = min_val + (max_val - min_val) * u
        """
        self.generator_active = True
        # Necessary constants
        modulus = (2 ** 31) - 1  # Really big modulus helps with uniformity
        multiplier = 66743
        offset = 1

        while True:
            if self.kill_generator:
                break

            self.seed = (self.seed * multiplier + offset) % modulus
            u = self.seed / modulus

            yield u

        self.generator_active = False