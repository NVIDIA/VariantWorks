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

from variantworks.io.baseio import BaseWriter
from variantworks.utils.metrics import convert_error_probability_arr_to_phred


class FastxWriter(BaseWriter):
    """Writer for FASTQ/FASTA files.

    Should be used with a context manager.
    """

    def __init__(self, output_path, mode):
        """Constructor VCFWriter class.

        Writes a FASTQ/FASTA records into a file using Biopython.

        Args:
            output_path : Output path for VCF output file.
            mode: Write mode for opening the output file.

        Returns:
            Instance of object.
        """
        super().__init__()
        self.output_path = output_path
        self.mode = mode
        self.file_obj = None

    def __enter__(self):
        """For contextmanager support."""
        self.file_obj = open(self.output_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For contextmanager support."""
        self.file_obj.close()

    def write_output(self, record_id, record_sequence, record_name=None, description=None, record_quality=None):
        """Write dataframe to FASTA or FASTQ in case record_quality is provided.

        Args:
            record_id : sequence record id.
            record_sequence : sequence data.
            record_name : sequence record name (optional).
            description : sequence record description (optional).
            record_quality : Corresponding records' sequence quality.
        """
        kargs = {
            'seq': Seq(record_sequence),
            'id': record_id,
        }
        if record_name:
            kargs['name'] = record_name
        if description:
            kargs['description'] = description
        record = SeqRecord(**kargs)
        if record_quality:
            record.letter_annotations["phred_quality"] = \
                convert_error_probability_arr_to_phred([1 - val for val in record_quality])
            SeqIO.write(record, self.file_obj, "fastq")
        else:
            SeqIO.write(record, self.file_obj, "fasta")
