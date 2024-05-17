from disparity_estimators.disparity_estimator import *

__disparity_estimator__ = {
    "softargmax": softargmax_estimator,
    "SME": single_modal_estimator,
    "DME": dominant_modal_estimator,
    "argmax": argmax_estimator,
}
