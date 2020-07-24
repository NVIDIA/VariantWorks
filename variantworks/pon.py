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

import cudf
import variantworks.merge_filter as mf


class PanelOfNormals:

    """
    A representation of a Panel-of-Normal samples.
    At its core, this is a pandas or cuDF dataframe.
    During merges, the object's <name> is prepended to its
    variables (along with the "pon" prefix) to prevent name
    collisions (i.e., "count" becomes "PON:<name>:count).

    A PanelOfNormals data member must have the following columns:
    - chrom
    - start_pos
    - end_pos
    - ref
    - alt

    And should have at least one of the following:
    - AF
    - count
    """

    def __init__(self):
        self.name = None
        self.source_file = None
        self.data = None

    def _compact(self):
        """
        Takes a dataframe from a VCF with multiple identical entries (like VCFs from COSMIC)
        and reduces the VCF to its core columns + a "count" column.
        """
        return

    def _prefix(self):
        return ":".join(["PON", self.name])

    def _tag(self, key):
        return ":".join([_prefix(self), key])

    def filter_by_allele_frequency(self, a, cutoff=0.02, dropAnnotations=True, dropNA=False, af_variable="AF"):
        #assert (cutoff <= 1.0 and cutoff >= 0.0)
        trim_columns = set(self.data.columns).difference(set(a.columns))
        merged_dfs = mf.merge_by_alleles(a, self.data, join="left")
        query_str = af_variable + " <= " + str(cutoff)
        if not dropNA:
            query_str += " or " + af_variable + " != " + af_variable
        filtered = merged_dfs.query(query_str)
        if dropAnnotations:
            filtered = filtered.drop(labels=list(trim_columns), axis=1)
        return filtered

    def filter_by_count(self, a, cutoff=1, dropAnnotations=True, dropNA=False, count_variable="count"):
        trim_columns = set(self.data.columns).difference(set(a.columns))
        merged_dfs = mf.merge_by_alleles(a, self.data, join="left")
        query_str = count_variable + " <= " + str(cutoff)
        if not dropNA:
            query_str += " or " + count_variable + " != " + count_variable
        filtered = merged_dfs.query(query_str)
        if dropAnnotations:
            filtered = filtered.drop(labels=list(trim_columns), axis=1)
        return filtered

    def filter_by_presence(self, a):
        return filter_by_count(self, a, cutoff=1)


def create_pon(df, pon_name="PON"):
    pon = PanelOfNormals()
    pon.name = pon_name
    pon.data = df
    return pon
