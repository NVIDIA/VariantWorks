#!/usr/bin/env python3

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

"""set setup.cfg file."""

import argparse
from collections import OrderedDict
import configparser
from datetime import datetime
from functools import partial
from io import StringIO
import json
import pathlib


def get_configuration_file_content(config_file_path):
    # Add a dummy section for comments outside sections
    with open(config_file_path, 'r') as f:
        config_string = '[dummy_comments_section]\n' + f.read()
    # Preserve in-section comments when updating
    # the file by reading them as keys with no value
    config = configparser.ConfigParser(
        comment_prefixes='', allow_no_value=True, strict=False)
    config.optionxform = str  # Preserve comments capitalization
    config.read_string(config_string)
    return config


def write_configuration_output(configuration_content, output_file_path):
    # Remove dummy section header and
    # write output to the configuration file
    config_file_obj = StringIO()
    configuration_content.write(config_file_obj)
    output_configuration = \
        config_file_obj.getvalue().split('\n', maxsplit=1)[1]
    with open(output_file_path, 'w') as fd:
        fd.write(output_configuration)


def update_configuration(file_path, fields):
    """Update each configration section."""
    config = get_configuration_file_content(file_path)
    for section, section_values in fields.items():
        if section not in config.sections():
            config.add_section(section)
        for key, value in section_values.items():
            config[section][key] = value
    write_configuration_output(config, file_path)


def add_nightly_version_suffix(file_path):
    config = get_configuration_file_content(file_path)
    config['metadata']['version'] = config['metadata']['version'] + '.dev' + datetime.today().strftime('%y%m%d')
    write_configuration_output(config, file_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update configuration file")
    parser.add_argument('--configuration_file',
                        help='path to setuptools configuration file',
                        required=True,
                        type=str)
    parser.add_argument('--fields',
                        help="json string formatted as"
                             " {'section_name': {key: value}}",
                        required=False,
                        default=None,
                        type=partial(
                            json.loads, object_pairs_hook=OrderedDict))
    parser.add_argument('--append-nightly-version-suffix',
                        action='store_true',
                        default=False,
                        help="Append nightly version suffix to current verion",
                        required=False)
    args = parser.parse_args()
    # Validate input configuration file existence
    input_conf_path = pathlib.Path(args.configuration_file)
    if not input_conf_path.is_file():
        raise FileNotFoundError(
            "Can not find input configuration file: {}".format(
                input_conf_path.resolve()))
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.fields:
        update_configuration(parsed_args.configuration_file, parsed_args.fields)
    if parsed_args.append_nightly_version_suffix:
        add_nightly_version_suffix(parsed_args.configuration_file)
