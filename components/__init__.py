"""Online adaptation components."""

from .online_vi import OnlineVI, BayesianConv2d, make_bayesian
from .active_learning import UncertaintyActiveLearner
from .drift_detection import ADWINDriftDetector
from .meta_learning import MAMLLearner
from .variance_recalibration import OnlineVarianceRecalibrator
