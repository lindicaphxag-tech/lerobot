#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Numeric feature schema invariance under JSON shape serialization."""

import json

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.feature_utils import (
    get_hf_features_from_features,
    validate_feature_numpy_array,
    validate_frame,
)


@pytest.mark.parametrize("shape", [(1,), (2,), (2, 3), (1, 2, 3), (4, 2, 2, 1)])
@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_json_roundtrip_preserves_numeric_validation(shape, dtype):
    original = {"observation.state": {"shape": shape, "dtype": dtype, "names": None}}
    decoded = json.loads(json.dumps(original))
    assert isinstance(decoded["observation.state"]["shape"], list)

    frame = {"task": "Move", "observation.state": np.zeros(shape, dtype=dtype)}
    validate_frame(frame, original)
    validate_frame(frame, decoded)
    assert decoded["observation.state"]["shape"] == list(shape)


@pytest.mark.parametrize("shape", [(1,), (2,), (2, 3), (3, 2, 1), (2, 2, 2, 2)])
def test_json_roundtrip_preserves_arrow_feature_types(shape):
    original = {"observation.state": {"shape": shape, "dtype": "float32", "names": None}}
    decoded = json.loads(json.dumps(original))
    before = get_hf_features_from_features(original)
    after = get_hf_features_from_features(decoded)
    assert before == after
    assert original["observation.state"]["shape"] == shape
    assert decoded["observation.state"]["shape"] == list(shape)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (1, 2, 3)])
def test_json_lists_still_reject_genuinely_incorrect_shapes(shape):
    feature = {"observation.state": {"shape": list(shape), "dtype": "float32", "names": None}}
    wrong = (shape[0] + 1, *shape[1:])
    frame = {"task": "Move", "observation.state": np.zeros(wrong, dtype="float32")}
    with pytest.raises(ValueError, match="expected shape"):
        validate_frame(frame, feature)


def test_dtype_mismatch_is_not_relaxed_by_list_shape_normalization():
    error = validate_feature_numpy_array("action", "float32", [2], np.ones(2, dtype=np.float64))
    assert "expected dtype" in error
    assert "expected shape" not in error


def test_dimensions_and_dtype_fail_independently():
    error = validate_feature_numpy_array("action", "float32", [2], np.ones((3,), dtype=np.float64))
    assert "expected dtype" in error
    assert "expected shape" in error
