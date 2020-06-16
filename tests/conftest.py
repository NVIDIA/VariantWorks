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

"""Share fixtures across multiple test fies."""

import bgzip
import os
import pytest
import subprocess
import tempfile


@pytest.fixture(scope='function')
def get_created_vcf_tabix_files():
    """Fixture for creating compressed vcf file and corresponding tabix file from bytes string.

    Returns:
        A function which creates these files
    """
    def created_vcf_tabix_files(vcf_content):
        _, tmp_input_path = tempfile.mkstemp(prefix='vw_test_file_', suffix='.vcf.gz')
        with open(tmp_input_path, 'wb') as raw_fd:
            with bgzip.BGZipWriter(raw_fd) as fh:
                fh.write(vcf_content)
        tabix_cmd_response = subprocess.run(['tabix', '-p', 'vcf', raw_fd.name])
        tabix_cmd_response.check_returncode()
        files_path = (raw_fd.name, raw_fd.name + ".tbi")
        created_files.append(files_path)
        return files_path
    created_files = list()
    yield created_vcf_tabix_files
    # cleanup
    try:
        for entry in created_files:
            os.remove(entry[0])
            os.remove(entry[1])
    except OSError as err:
        raise type(err)('Can not remove input files: {}, {}'.format(*entry)) from err
