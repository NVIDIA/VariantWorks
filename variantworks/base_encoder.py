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
"""Classes and function to encode bases in a read."""


def rgb_to_hex(c):
    """Concert rgb tuple/list into Hex."""
    return "#{0:02x}{1:02x}{2:02x}".format(*c)


base_color_decoder = {
    '\0': [255, 255, 255],      # white (null char for cells initiated to 'zero' )
    'A': [0, 128, 0],           # green
    'T': [255, 0, 0],           # red
    'C': [0, 0, 255],           # blue
    'G': [255, 255, 0],         # yellow
    'N': [0, 0, 0]              # black
}
