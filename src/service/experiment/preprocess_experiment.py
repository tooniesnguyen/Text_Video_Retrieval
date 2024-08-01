from src.service.abstraction import Experiments, Predictor
from src.utils.timer import time_complexity
from src.utils.logger import Logger
import traceback


class ExperimentsImpl(Experiments):
    def __init__(self, predictor: Predictor, logger: Logger) -> None:
        self.logger = logger.get_tracking(__name__)
        self.predictor = predictor
    
    @time_complexity(name_process='PHASE EXPERIMENT')
    def run(self):

        try:
            self.logger.info('Start predicting !')
            self.predictor.predict()

        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc())