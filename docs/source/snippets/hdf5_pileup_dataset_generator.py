#!/usr/bin/env python
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
"""Code snippet for HDF5 Pileup Dataset Generator."""

import h5py
import numpy as np
import os
import pathlib
import tempfile

from variantworks.encoders import PileupEncoder, ZygosityLabelEncoder
from variantworks.io.vcfio import VCFReader

# Get VariantWorks root directory
repo_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.absolute()

# Get BAM and VCF files for the raw sample data.
data_folder = os.path.join(repo_root_dir, "tests", "data")
bam = os.path.join(data_folder, "small_bam.bam")
samples = os.path.join(data_folder, "candidates.vcf.gz")

# Generate the variant entries using VCF reader.
vcf_reader = VCFReader(vcf=samples, bams=[bam], is_fp=False)
print("Serializing {} entries...".format(len(vcf_reader)))

# Setup encoder for samples and labels.
sample_encoder = PileupEncoder(window_size=100, max_reads=100, layers=[
    PileupEncoder.Layer.READ])
label_encoder = ZygosityLabelEncoder()

# Create HDF5 datasets.
_, output_file = tempfile.mkstemp(prefix='hdf5_generation_snippet_', suffix=".hdf5")
h5_file = h5py.File(output_file, "w")
encoded_data = h5_file.create_dataset("encodings",
                                      shape=(len(vcf_reader), sample_encoder.depth,
                                             sample_encoder.height, sample_encoder.width),
                                      dtype=np.float32, fillvalue=0)
label_data = h5_file.create_dataset("labels",
                                    shape=(len(vcf_reader),), dtype=np.int64, fillvalue=0)

# Loop through all entries, encode them and save them in HDF5.
for i, variant in enumerate(vcf_reader):
    encoding = sample_encoder(variant)
    label = label_encoder(variant)
    encoded_data[i] = encoding
    label_data[i] = label

# Close HDF5 file.
h5_file.close()

# Remove output file.
os.remove(output_file)
