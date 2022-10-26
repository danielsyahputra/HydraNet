import numpy as np

class Transform():
    def __init__(self) -> None:
        self.min_age = 1
        self.max_age = 116
        self.max_age_log = 2.0644579892269186

    def get_normalize(self, original_value):
        return (original_value - self.min_age) / (self.max_age - self.min_age)

    def get_log_age(self, original_value):
        return np.log10(original_value) / self.max_age_log
    
    def get_original_from_log(self, log_age_value):
        return np.exp(log_age_value) * self.max_age_log

    def get_original_age(self, normalized_value):
        return normalized_value * (self.max_age - self.min_age) + self.min_age