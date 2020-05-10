#Abstract and implementation clases for label loaders.

import vcf

from claragenomics.variantworks.types import VariantZygosity, Variant

class LabelLoaderIterator():
    def __init__(self, label_loader):
        assert(isinstance(label_loader, BaseLabelLoader))
        self._label_loader = label_loader
        self._index = 0

    def __next__(self):
        if (self._index < len(self._label_loader)):
            result = self._label_loader[self._index]
            self._index += 1
            return result
        raise StopIteration

class BaseLabelLoader():
    def __init__(self, allow_snps=True, allow_multiallele=True, allow_multisample=False):
        """Base class label loader that sotres variant filters and implements indexing
        and length methods.
        """
        self._allow_snps = allow_snps
        self._allow_multiallele = allow_multiallele
        self._allow_multisample = allow_multisample
        self._labels = []

    def __getitem__(self, idx):
        return self._labels[idx]

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        return LabelLoaderIterator(self)

class VCFLabelLoader(BaseLabelLoader):
    """VCF based label loader for true and false positive example files.
    """
    def __init__(self, tp_vcfs, fp_vcfs, **kwargs):
        super().__init__(**kwargs)

        for tp_vcf in tp_vcfs:
            self._parse_vcf(tp_vcf, self._labels)
        for fp_vcf in fp_vcfs:
            self._parse_vcf(fp_vcf, self._labels, is_fp=True)

    def _get_variant_type(self, record, is_fp=False):
        """Determine variant type from pyvcf record.
        """
        if is_fp:
            return VariantZygosity.NO_VARIANT
        if record.num_het > 0:
            return VariantZygosity.HETEROZYGOUS
        elif record.num_hom_alt > 0:
            return VariantZygosity.HOMOZYGOUS
        assert(False), "Unexpected variant type - {}".format(record)

    def _parse_vcf(self, vcf_file, labels, is_fp=False):
        """Parse VCF file and retain labels after they have passed filters.
        """
        assert(vcf_file[-3:] == ".gz"), "VCF file needs to be compressed and indexed" # Check for compressed file
        vcf_reader = vcf.Reader(open(vcf_file, "rb"))
        for record in vcf_reader:
            if (not(self._allow_snps and record.is_snp)):
                continue
            if (not self._allow_multisample and record.num_called > 1):
                continue
            if (not self._allow_multiallele and len(record.ALT) > 1):
                continue
            chrom = record.CHROM
            pos = record.POS
            ref = record.REF
            var_type = self._get_variant_type(record, is_fp)
            for alt in record.ALT:
                var_allele = alt.sequence

            labels.append(Variant(chrom, pos, ref, var_type, var_allele))
