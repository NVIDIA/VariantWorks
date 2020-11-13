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
"""Classes for reading and writing BED files."""

import pandas as pd

from variantworks.types import BEDEntry
from variantworks.io.baseio import BaseReader


class BEDReader(BaseReader):
    """Reader for BEDPE files."""

    def __init__(self, bed_path):
        """Constructor BEDPEReader class.

        Reads BEDPE entries from a BEDPE file.

        Args:
            bed_path: Path to BEDPE file.

        Returns:
            Instance of object.
        """
        super().__init__()
        self._bed_path = bed_path
        self._df = pd.read_csv(self._bed_path, delimiter="\t")

    def dataframe(self):
        """Return dataframe object for file."""
        return self._df

    def __len__(self):
        """Return number of entries in file."""
        return len(self._df)

    def __getitem__(self, idx):
        """Return a BEDPE entry."""
        row = self._df.iloc[idx].to_dict()
        return BEDEntry(row)
