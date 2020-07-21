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
"""Classes and algorithms for variant consensus generation."""


class MajorityConsensus():
    """A class for consensus caller based on simple majority of callers."""

    def __init__(self, variant_readers, caller_cols, min_callers=2):
        """Constructor for MajorityConsensus caller.

        This algorithm generate a consensus call by looking at the caller support
        for each variant, and evaluates that against the minimum support requested
        by the user. Variants that meet the minimum requirement are kept in the final
        call set.

        Args:
            variant_readers : List of VariantWorks variant reader objects such as VCFReader
            caller_cols : List of column names across variant reader objects that denote
                          the columns identifying various callers
            min_callers : Minimum caller support for each variant

        Returns:
            A new dataframe with consensus called variants.
        """
        if (len(variant_readers) < 2):
            raise RuntimeError("Need at least 2 callers for {}".format(self.__class__.__name__))
        self._variant_readers = variant_readers
        self._caller_cols = caller_cols
        self._min_callers = min_callers
        self._common_cols = ["chrom", "start_pos", "ref", "alt"]

    def generate_consensus(self):
        """Run the consensus calling algorithm."""
        df_consensus = self._variant_readers[0].df[self._common_cols + self._variant_readers[0]._tags]
        for i in range(1, len(self._variant_readers)):
            df_new = self._variant_readers[i].df[self._common_cols + self._variant_readers[i]._tags]
            df_consensus = df_consensus.merge(df_new)
        df_consensus["caller_count"] = df_consensus[self._caller_cols].sum(axis=1)
        df_consensus = df_consensus[df_consensus["caller_count"] >= self._min_callers]
        return df_consensus
