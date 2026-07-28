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
"""Packaging metadata contracts for NeMo workspace integration."""

from pathlib import Path

import pytest
import tomllib


CACHED_DEPENDENCIES_TRANSFORMERS_REQUIREMENT = "transformers>=5.8.1,<5.9.0"


@pytest.mark.unit
def test_transformers_requirement_matches_nemo_workspace_cache_contract() -> None:
    """Keep Bridge metadata compatible with NeMo's cached workspace dependency list."""
    repository_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repository_root / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    transformers_requirement = next(dependency for dependency in dependencies if dependency.startswith("transformers"))

    assert transformers_requirement == CACHED_DEPENDENCIES_TRANSFORMERS_REQUIREMENT
