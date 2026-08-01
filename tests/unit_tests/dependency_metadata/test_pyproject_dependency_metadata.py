# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_fast_hadamard_transform_has_lock_time_metadata() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    uv_configuration = pyproject["tool"]["uv"]
    metadata_entries = uv_configuration["dependency-metadata"]
    fast_hadamard_metadata = next(
        (
            entry
            for entry in metadata_entries
            if entry["name"] == "fast-hadamard-transform"
        ),
        None,
    )

    assert fast_hadamard_metadata == {
        "name": "fast-hadamard-transform",
        "version": "1.1.0",
        "requires-dist": ["torch", "packaging", "ninja"],
    }
    assert (
        "fast-hadamard-transform @ "
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0"
        in uv_configuration["override-dependencies"]
    )
