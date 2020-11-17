#
# Copyright 2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Metrics Utilities."""

import numpy as np


def convert_error_probability_arr_to_phred(err_prob_arr, eps=np.power(10, -9.3)):
    """Convert error probability rate to quality (Phred) scores.

    Args:
        err_prob_arr : Array with error probabilities
        eps : Epsilon error value to use for stability when error is zero. [QV93]

    Returns:
        Array with error values converted to PHRED score.
    """
    if any(i < 0 or i > 1 for i in err_prob_arr):
        raise ValueError("all values in error probability array must be between 0 and 1")
    # replace zeros with eps (when softmax is REALLY sure)
    err_prob_arr = np.array(err_prob_arr)
    err_prob_arr[err_prob_arr < eps] = eps
    return np.trunc(-10 * np.log10(err_prob_arr))
