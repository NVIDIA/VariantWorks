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

"""Contains mocked file object inputs for tests."""

import bgzip
import io
import os
import pytest
import subprocess
import tempfile


def mock_file_input():
    """Return a string stream of an unfiltered vcf file content."""
    return b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED
1	139098	.	CT	T	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139295	.	G	AC	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139738	.	G	C,A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	139988	.	T	A	50	.	DP=34;AF=0.0194118	GT:GQ	0/1:50
1	139994	.	G	C	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140009	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140013	.	C	A	50	.	DP=35;AF=0.0185714	GT:GQ	0/1:50
1	140016	.	T	C	50	.	DP=34;AF=0.0194118	GT:GQ	1:50
1	240021	.	T	C	50	.	DP=34;AF=0.0294118	GT:GQ	1:50
1	240023	.	A	G	50	.	DP=35;AF=0.0285714	GT:GQ	1:50
1	240046	.	C	A	50	.	DP=34;AF=0.0294118	GT:GQ	1:50
1	240090	.	T	A	50	.	DP=22;AF=0.0454545	GT:GQ	1:50
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	1:50
1	240154	.	T	C	50	.	DP=13;AF=0.0769231	GT:GQ	1:50
"""


def mock_invalid_file_input():
    """Returns a string stream of a vcf file content which is supposed to raise a RuntimeError.

    More than one called sample
    """
    return b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED  CALLED2
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	1:50    1/1:50
1	240154	.	T	C	50	.	DP=13;AF=0.0769231	GT:GQ	1:50    0/1:50
"""


def mock_small_filtered_file_input():
    """Return string stream of small filtered vcf content."""
    return b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	CALLED
1	139861	.	T	A	50	.	DP=15;AF=0.0666667	GT:GQ	0/1:50
1	139976	.	G	A	50	.	DP=35;AF=0.0185714	GT:GQ	1/1:50
1	240147	.	C	T	50	.	DP=13;AF=0.692308	GT:GQ	0/1:50
"""


def created_vcf_tabix_files(content):
    _, tmp_input_path = tempfile.mkstemp(prefix='vw_test_file_', suffix='.vcf.gz')
    with open(tmp_input_path, 'wb') as raw_fd:
        with bgzip.BGZipWriter(raw_fd) as fh:
            fh.write(content)
    tabix_cmd_response = subprocess.run(['tabix', '-p', 'vcf', raw_fd.name])
    tabix_cmd_response.check_returncode()
    return raw_fd.name, raw_fd.name + ".tbi"


@pytest.fixture(scope='function')
def get_created_vcf_tabix_files(request):
    vcf_path, tabix_path = created_vcf_tabix_files(request.param)
    yield vcf_path, tabix_path
    # cleanup
    try:
        os.remove(vcf_path)
        os.remove(tabix_path)
    except OSError as err:
        raise type(err)('Can not remove input files') from err


# class MockPyVCFReader:
#     """Return VCFReader instance with mocked file content."""
#
#     class ContentType(Enum):
#         """VCF file content type for mocking."""
#         UNFILTERED = 0
#         INVALID = 1
#         SMALL_FILTERED = 2
#
#     original_vcfeader_get_file_reader_method = VCFReader._get_file_reader
#
#     @staticmethod
#     def _get_unfiltered_vcf_reader(*args, **kargs):
#         return MockPyVCFReader.original_vcfeader_get_file_reader_method(mock_file_input())
#
#     @staticmethod
#     def _get_invalid_vcf_reader(*args, **kargs):
#         return MockPyVCFReader.original_vcfeader_get_file_reader_method(mock_invalid_file_input())
#
#     @staticmethod
#     def _get_small_filtered_vcf_reader(*args, **kargs):
#         return MockPyVCFReader.original_vcfeader_get_file_reader_method(mock_small_filtered_file_input())
#
#     _content_type_to_mocked_reader_method = {
#         ContentType.UNFILTERED:         _get_unfiltered_vcf_reader.__func__,
#         ContentType.INVALID:            _get_invalid_vcf_reader.__func__,
#         ContentType.SMALL_FILTERED:     _get_small_filtered_vcf_reader.__func__,
#     }
#
#     @staticmethod
#     def get_reader(mp, vcf_bam_list, content_type):
#         """Mock VCFReader reader content according to given content type.
#
#         Args:
#               mp: Pytest monkeypatch context
#               vcf_bam_list: List of VcfBamPath objects
#               content_type: Type of request vcf content
#
#         Returns:
#               VCF content as StringIO
#         """
#         with mp.context() as m:
#             m.setattr(VCFReader, "_get_file_reader",
#                       MockPyVCFReader._content_type_to_mocked_reader_method[content_type])
#             vcf_loader = VCFReader(vcf_bam_list)
#         return vcf_loader
