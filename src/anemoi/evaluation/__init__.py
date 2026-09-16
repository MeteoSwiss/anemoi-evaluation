"""In-memory forecast evaluation for anemoi models."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from anemoi.evaluation import binning
from anemoi.evaluation import metrics
from anemoi.evaluation import regions
from anemoi.evaluation import statistics
from anemoi.evaluation import weights
from anemoi.evaluation.aggregation import AggregationState
from anemoi.evaluation.aggregation import Aggregator
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.evaluate import init_times
from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.output import load_state
from anemoi.evaluation.output import merge
from anemoi.evaluation.output import to_xarray
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.base import ClimatologySourceBase
from anemoi.evaluation.sources.base import ForecastSourceBase
from anemoi.evaluation.sources.base import TargetSourceBase
from anemoi.evaluation.sources.climatology import ArrayClimatology
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource
from anemoi.evaluation.sources.persistence import PersistenceForecastSource

try:
    __version__ = version("anemoi-evaluation")
except PackageNotFoundError:  # a source tree on the path rather than an installed package
    __version__ = "0+unknown"

__all__ = [
    "AggregationState",
    "Aggregator",
    "ArrayClimatology",
    "ArrayTargets",
    "ClimatologySourceBase",
    "DatasetTargets",
    "Evaluation",
    "FakeForecastSource",
    "ForecastSourceBase",
    "Frame",
    "Grid",
    "InferenceForecastSource",
    "PersistenceForecastSource",
    "TargetSourceBase",
    "__version__",
    "binning",
    "init_times",
    "load_state",
    "merge",
    "metrics",
    "regions",
    "statistics",
    "to_xarray",
    "weights",
]
