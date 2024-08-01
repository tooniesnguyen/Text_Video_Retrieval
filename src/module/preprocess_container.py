import os
from pathlib import Path
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from thespian.actors import ActorSystem

from src.utils.constants import CONFIG_FILE
from src.service.abstraction import Experiments, Predictor


from src.service.predictor.preprocess_predictor import PredictorImpl
from src.service.experiment.preprocess_experiment import ExperimentsImpl
# from src.service.data_preprocess.Transnetv2 import TransNetV2Implement
from src.service.data_preprocess.Autoshot import AutoShotImplement

from src.utils.logger import Logger

FILE = Path(__file__).resolve()
WORK_DIR = FILE.parents[2]

class ApplicationContainer(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(modules=["src.service.experiment.preprocess_experiment"])

    service_config = providers.Configuration()
    service_config.from_yaml(filepath=os.path.join(WORK_DIR, CONFIG_FILE))
    actor_system = providers.Singleton(ActorSystem)

    logger = providers.Singleton(
        Logger,
        log_dir=service_config.logger.log_dir,
        log_clear_days=service_config.logger.log_clear_days
    )

    # transnetv2 = providers.Singleton(
    #     TransNetV2Implement
    # )

    autoshot = providers.Singleton(
        AutoShotImplement
    )

    predictor = providers.AbstractSingleton(Predictor)
    predictor.override(
        providers.Singleton(
            PredictorImpl,
            AutoShotImplement,
            service_config.data.path_video,
            service_config.data.path_img,

        )
    )

    experiments = providers.AbstractSingleton(Experiments)
    experiments.override(
        providers.Singleton(
            ExperimentsImpl,
            predictor = predictor,
            logger = logger
        )
    )