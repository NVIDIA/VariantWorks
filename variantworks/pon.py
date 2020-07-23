import cudf
import variantworks.merge_filter as mf


class PanelOfNormals:

    def __init__(self):
        self.name = None
        self.source_file = None
        self.data = None

    def _compact(self):
        """
        Takes a dataframe from a VCF with multiple identical entries (like VCFs from COSMIC)
        and reduces the VCF to its core columns + a "count" column.
        """

    def filter_by_allele_frequency(self, a, cutoff=0.02, af_variable="AF"):
        assert(cutoff <= 1.0 and cutoff >= 0.0)
        merged_dfs = mf.merge_by_alleles(a, self.data, join="left")
        query_str = af_variable + " <= " + str(af_cutoff)
        return merged_dfs.query(query_str)

    def filter_by_count(self, a, cutoff=1, count_variable="count"):
        merged_dfs = mf.merge_by_alleles(a, self.data, join="left")
        query_str = count_variable + " <= " + str(cutoff)
        return merged_dfs.query(query_str)

    def filter_by_presence(self, a):
        return filter_by_count(self, a, cutoff=1)


def create_pon(pon_file, pon_name):
    return
