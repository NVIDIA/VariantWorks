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
"""Classes for reading and writing FASTQ files."""

from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import itertools

from variantworks.io.baseio import BaseWriter
from variantworks.utils.metrics import convert_error_probability_arr_to_phred


class FastqWriter(BaseWriter):
    """Writer for FASTQ files."""

    def __init__(self, output_path, records_ids, records_seqs, records_qualities):
        """Constructor VCFWriter class.

        Writes a FASTQ records into a file using Biopython.

        Args:
            output_path : Output path for VCF output file.
            records_ids : List of records' id numbers.
            records_seqs : Corresponding list with records' sequence literals.
            records_qualities : Corresponding records' list of lists with each nucleotide certainty score.

        Returns:
            Instance of object.
        """
        super().__init__()
        self.output_path = output_path
        self.records_ids = records_ids
        self.records_seqs = records_seqs
        self.records_qualities = records_qualities

    def write_output(self):
        """Write dataframe to VCF."""
        output_records = list()
        for name, seq, q_score in itertools.zip_longest(self.records_ids, self.records_seqs, self.records_qualities):
            record = SeqRecord(Seq(seq),
                               id=name,
                               description="Generated consensus sequence by NVIDIA VariantWorks")
            record.letter_annotations["phred_quality"] = \
                convert_error_probability_arr_to_phred([1 - val for val in q_score])
            output_records.append(record)

        with open(self.output_path, "w") as fd:
            SeqIO.write(output_records, fd, "fastq")
