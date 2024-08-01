from numpy import ndarray
from src.service.abstraction import Predictor
from src.utils.logger import Logger
from src.utils.timer import time_complexity


class PredictorImpl(Predictor):
    def __init__(self, model, input_dir: str, output_dir: str) -> None:
        self.model = model(input_dir, output_dir)

    @time_complexity(name_process='PHASE TEST')
    def predict(self) :
        self.model.run()
        