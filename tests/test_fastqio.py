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

from Bio import SeqIO
import itertools
import numpy as np
import os
import pytest
import tempfile

from variantworks.io.fastxio import FastxWriter
from variantworks.utils.metrics import convert_error_probability_arr_to_phred


@pytest.mark.parametrize(
    'err_prob_values,err_prob_values_with_exception',
    [
        ([0.23, 0.134, 0.8, 0.026, 0.0011, 0.00012, 0.00001], [0.23, 0.134, 0.8, 0.026, 0.0011, 0.00012, 0.00001, 20]),
    ]
)
def test_err_prob_to_phred_score(err_prob_values, err_prob_values_with_exception):
    """Test conversion between error probability to Phred quality score."""
    output = convert_error_probability_arr_to_phred(err_prob_values)
    assert np.array_equal(output, [6, 8, 0, 15, 29, 39, 50])
    with pytest.raises(ValueError):
        convert_error_probability_arr_to_phred(err_prob_values_with_exception)


def test_fastq_writer():
    """Test FASTQ writer"""
    nucleotides_sequences = ["ACTGATG",
                             "ACAC"]
    nucleotides_certainties = [[0.87, 0.424, 0.625, 0.99, 0.001, 0.64, 0.787],
                               [0.24, 0.53, 0.765, 0.362]]
    record_ids = ["dl_seq_1",
                  "dl_seq_2"]

    _, file_path = tempfile.mkstemp(prefix='vw_test_fastq_writer_', suffix='.fastq')
    try:
        with FastxWriter(file_path, 'w') as fqout:
            for record_id, seq, q_score in \
                    itertools.zip_longest(record_ids, nucleotides_sequences, nucleotides_certainties):
                fqout.write_output(record_id, seq, q_score)
        # verify created FASTQ file values
        for idx, record in enumerate(SeqIO.parse(file_path, "fastq")):
            assert record_ids[idx] == record.id
            assert nucleotides_sequences[idx] == record.seq
            assert np.array_equal(
                # convert certainties to error rates
                convert_error_probability_arr_to_phred([1 - val for val in nucleotides_certainties[idx]]),
                record.letter_annotations["phred_quality"]
            )
    finally:
        os.remove(file_path)


def test_fasta_writer():
    """Test FASTA writer"""
    nucleotides_sequences = ["ACTGATG",
                             "ACAC"]

    record_ids = ["dl_seq_1",
                  "dl_seq_2"]

    _, file_path = tempfile.mkstemp(prefix='vw_test_fastq_writer_', suffix='.fasta')
    try:
        with FastxWriter(file_path, 'w') as fqout:
            for record_id, seq in itertools.zip_longest(record_ids, nucleotides_sequences):
                fqout.write_output(record_id, seq)
        # verify created FASTA file values
        for idx, record in enumerate(SeqIO.parse(file_path, "fasta")):
            assert record_ids[idx] == record.id
            assert nucleotides_sequences[idx] == record.seq
    finally:
        os.remove(file_path)
