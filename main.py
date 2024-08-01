import warnings
warnings.filterwarnings("ignore")

from dependency_injector.wiring import Provide, inject
from src.utils.debugger import pretty_errors
from src.module.preprocess_container import ApplicationContainer
from src.service.experiment.preprocess_experiment import ExperimentsImpl
from threadpoolctl import threadpool_limits, threadpool_info

def create_container(environment: str):
    container = ApplicationContainer()
    container.wire(modules=[environment])

@inject
def run(experiments: ExperimentsImpl = Provide[ApplicationContainer.experiments]) -> None:
    experiments.run()

create_container(environment=__name__)

if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='openmp'):
        run()