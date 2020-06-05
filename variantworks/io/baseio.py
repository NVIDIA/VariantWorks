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
"""Base class readers and writers."""

from abc import ABC, abstractmethod


class BaseReaderIterator:
    """Iterator class for VCF reader."""

    def __init__(self, reader):
        assert(isinstance(reader, BaseReader))
        self._reader = reader
        self._index = 0

    def __next__(self):
        if self._index < len(self._reader):
            result = self._reader[self._index]
            self._index += 1
            return result
        raise StopIteration


class BaseReader(ABC):
    """Base class for format reader."""

    def __init__(self):
        """Constructor."""

        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Index into reader class to fetch entry."""

        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Returns total number of entries in reader."""

        return NotImplementedError

    def __iter__(self):
        return BaseReaderIterator(self)
