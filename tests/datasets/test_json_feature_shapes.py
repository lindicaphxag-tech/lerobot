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
"""LeRobot feature shapes must remain valid through JSON dataset metadata."""
import json

import datasets
import numpy as np
import pytest

from lerobot.datasets.feature_utils import (
    get_hf_features_from_features,
    validate_feature_numpy_array,
    validate_frame,
)


@pytest.mark.parametrize("shape", [(2,), [2], (2, 3), [2, 3], (1,), [1]])
def test_numpy_feature_accepts_identically_sized_list_and_tuple_shapes(shape):
    frame = np.zeros(shape, dtype=np.float32)
    assert validate_feature_numpy_array("observation.state", "float32", shape, frame) == ""


def test_wrong_dimensions_and_dtype_still_fail_after_json_normalization():
    assert "expected shape" in validate_feature_numpy_array(
        "observation.state", "float32", [2, 3], np.zeros((3, 2), dtype=np.float32)
    )
    assert "expected dtype" in validate_feature_numpy_array(
        "observation.state", "float32", [2, 3], np.zeros((2, 3), dtype=np.float64)
    )


def test_validate_frame_roundtripped_metadata_list_shape_matches_ndarray_tuple():
    features = json.loads(json.dumps({
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (2, 3), "names": None},
    }))
    frame = {
        "task": "move",
        "observation.state": np.zeros((6,), dtype=np.float32),
        "action": np.zeros((2, 3), dtype=np.float32),
    }
    validate_frame(frame, features)

    with pytest.raises(ValueError, match="expected shape"):
        validate_frame({**frame, "action": np.zeros((3, 2), dtype=np.float32)}, features)


@pytest.mark.parametrize("declared", [[1], (1,)])
def test_singleton_feature_preserves_hf_scalar_semantics_after_json(declared):
    schema = get_hf_features_from_features({
        "reward": {"dtype": "float32", "shape": declared, "names": None}
    })
    assert isinstance(schema["reward"], datasets.Value)


def test_higher_dimensional_json_shape_still_uses_array_feature():
    schema = get_hf_features_from_features({
        "observation.state": {"dtype": "float32", "shape": [2, 3], "names": None}
    })
    assert isinstance(schema["observation.state"], datasets.Array2D)
