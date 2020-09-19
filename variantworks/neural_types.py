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
"""Custom neural types for VariantWorks."""

from nemo.core.neural_types import LabelsType, ChannelType


class ReadPileupNeuralType(ChannelType):
    """Element type to represent a variant encoding."""


class VariantZygosityNeuralType(LabelsType):
    """Element type to represent variant type (no variant, heterozygous or homozygous."""


class SummaryPileupNeuralType(ChannelType):
    """Element type to represent a consensus summary encoding."""


class HaploidNeuralType(LabelsType):
    """Element type to represent label output from HaploidLabelEncoder."""
