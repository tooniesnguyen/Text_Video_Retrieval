from abc import abstractmethod, ABC
import numpy as np

class Predictor(ABC):

    @abstractmethod
    def predict(self) -> None:
        pass

