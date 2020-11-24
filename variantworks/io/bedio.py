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

from enum import Enum
import pandas as pd
import itertools
from collections import Counter  # Counter counts the number of occurrences of each item

from variantworks.types import BEDEntry
from variantworks.io.baseio import BaseReader


class BEDReader(BaseReader):
    """Reader for BEDPE files."""

    # Supported BED types.
    class BEDType(Enum):
        """An enum definining supported BED types."""

        BED = 0
        BEDPE = 1

    def __init__(self, bed_path, bed_type, header_is_comment=False):
        """Constructor BEDPEReader class.

        Reads BEDPE entries from a BEDPE file.

        Args:
            bed_path: Path to BEDPE file.
            bed_type: Type of BED file (BEDType.BED or BEDType.BEDPE)
            header_is_comment: Whether header prefixed by the comment character '#'? (default: False)

        Returns:
            Instance of object.
        """
        super().__init__()
        self._bed_path = bed_path
        assert(isinstance(bed_type, self.BEDType)), "bed_type must be of BEDType enum."
        self._bed_type = bed_type
        self._header_is_comment = header_is_comment
        if self._header_is_comment is True:
            self._df = self._read_data(self._bed_path)
        else:
            self._df = pd.read_csv(self._bed_path, delimiter="\t")
        self._enforce_bed_types()

    # borrowed from: https://stackoverflow.com/a/36772979 CC BY-SA 3.0
    def _read_data(self, path):
        """Manually reads in column names if they are behind a comment."""
        with open(path) as handle:
            *_comments, names = itertools.takewhile(
                lambda line: line.startswith('#'), handle)

            # This is not the most robust way, adjust for your needs :)
            names = names[2:].rstrip().split("\t")
            self._uniquify(names)
            return pd.read_table(path, header=None, names=names, comment='#', na_values='_')

    # borrowed from: https://stackoverflow.com/a/30651963 CC BY-SA 3.0
    def _uniquify(self, seq, suffs=itertools.count(1)):
        """Add uniquifying suffix to duplicated strings in a list.

        Make all the items unique by adding a suffix (1, 2, etc).

        Args:
            seq: is mutable sequence of strings.
            suffs: is an optional alternative suffix iterable.
        """
        not_unique = [k for k, v in Counter(seq).items() if v > 1]  # so we have: ['name', 'zip']
        # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
        suff_gens = dict(zip(not_unique, itertools.tee(suffs, len(not_unique))))
        for idx, s in enumerate(seq):
            try:
                suffix = str(next(suff_gens[s]))
            except KeyError:
                # s was unique
                continue
            else:
                seq[idx] += suffix

    def _enforce_bed_types(self):
        if self._bed_type == self.BEDType.BED or self._bed_type == self.BEDType.BEDPE:
            self._df.iloc[:, [0]] = self._df.iloc[:, [0]].astype('object')  # chrom1
            self._df.iloc[:, [1]] = self._df.iloc[:, [1]].astype('int64')  # start1
            self._df.iloc[:, [2]] = self._df.iloc[:, [2]].astype('int64')  # end1
        if self._bed_type == self.BEDType.BEDPE:
            self._df.iloc[:, [3]] = self._df.iloc[:, [3]].astype('object')  # chrom2
            self._df.iloc[:, [4]] = self._df.iloc[:, [4]].astype('int64')  # start2
            self._df.iloc[:, [5]] = self._df.iloc[:, [5]].astype('int64')  # end2

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
