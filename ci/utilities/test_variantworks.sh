#!/bin/bash

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


logger "Install VariantWorks external dependencies..."
python -m pip install -r requirements.txt

logger "Install pyclaragenomics..."
python -m pip install -e .

logger "Run Tests..."
python -m pytest -s tests/

logger "Run Documentation Snippets..."
# Reverse alphabetical order, so the training snippet will be executed before inference
for f in $(find docs/source/snippets/*.py | sort -r); do
  logger "Executing \"${f}\""
  python "${f}"
done
