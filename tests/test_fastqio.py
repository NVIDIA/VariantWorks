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

import numpy as np
import pytest

from variantworks.io.fastqio import FastqWriter


@pytest.mark.parametrize(
    'err_prob_values,err_prob_values_with_exception',
    [
        ([0.23, 0.134, 0.8, 0.026, 0.0011, 0.00012, 0.00001], [0.23, 0.134, 0.8, 0.026, 0.0011, 0.00012, 0.00001, 20]),
    ]
)
def test_err_prob_to_phred_score(err_prob_values, err_prob_values_with_exception):
    """Test conversion between err prob to phred score."""
    output1 = FastqWriter._convert_error_probability_arr_to_phred(err_prob_values)
    assert np.array_equal(output1, [6, 8, 0, 15, 29, 39, 50])
    with pytest.raises(ValueError):
        FastqWriter._convert_error_probability_arr_to_phred(err_prob_values_with_exception)
