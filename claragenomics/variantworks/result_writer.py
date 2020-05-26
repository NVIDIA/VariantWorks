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

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp
import vcf
from vcf.parser import _Format, _Substitution, _Call
from vcf.model import make_calldata_tuple

from claragenomics.variantworks.types import VariantZygosity


class ResultWriter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def write_output(self):
        pass


class VCFResultWriter(ResultWriter):

    zygosity_to_vcf_genotype = {
        VariantZygosity.NO_VARIANT:     "0/0",
        VariantZygosity.HETEROZYGOUS:   "0/1",
        VariantZygosity.HOMOZYGOUS:     "1/1",
    }

    def __init__(self, variant_label_loader, infered_zygosities, output_location=None):
        self.vcf_path_to_reader_writer = dict()
        self.variant_label_loader = variant_label_loader
        self.infered_zygosities = infered_zygosities
        self.output_location = Path(output_location) if output_location else Path(mkdtemp())

    def _get_file_writer(self, variant):
        vcf_file_path = Path(variant.vcf)
        if vcf_file_path not in self.vcf_path_to_reader_writer:
            vcf_reader = vcf.Reader(filename=str(vcf_file_path))
            vcf_reader.formats["IR"] = _Format("IR", 1, "String", "Infered Results")
            vcf_writer = vcf.Writer(open(self.output_location / (vcf_file_path.name + '.vcf'), 'w'), vcf_reader)
            self.vcf_path_to_reader_writer[vcf_file_path] = (vcf_reader, vcf_writer)
        return self.vcf_path_to_reader_writer[vcf_file_path]

    def _get_encoded_zygosity_to_genotype(self, idx):
        return VCFResultWriter.zygosity_to_vcf_genotype[self.infered_zygosities[idx]]

    def write_output(self):
        # Iterate over all variances
        for variant in self.variant_label_loader:
            file_reader, file_writer = self._get_file_writer(variant)
            for record in file_reader.fetch(variant.chrom, int(variant.pos)-1, variant.pos):
                if variant.ref != record.REF or variant.allele not in record.ALT:
                    raise RuntimeError("")
                record.ALT = [_Substitution(variant.allele)]
                record.FORMAT += ':IR'
                new_samples = list()
                for s in record.samples:
                    new_sample_data = \
                        [str(elem) if elem is not None else "." for elem in s.data] +\
                        [self._get_encoded_zygosity_to_genotype(variant.idx)]
                    sample_format = make_calldata_tuple(record.FORMAT.split(":"))
                    new_samples.append(_Call(record, s.sample, sample_format(*new_sample_data)))
                record.samples = new_samples
                file_writer.write_record(record)
        # Close all file writers
        for _, fbuffers in self.vcf_path_to_reader_writer.items():
            fbuffers[1].close()
