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

"""Mixed-precision constant-coordinate normalization regression (#4973)."""

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import NormalizerProcessorStep, TransitionKey, UnnormalizerProcessorStep


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mode",
    [
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ],
)
@pytest.mark.parametrize("feature_type", [FeatureType.STATE, FeatureType.ACTION])
def test_constant_coordinate_finite_for_all_modes_dtypes_and_input_paths(dtype, mode, feature_type):
    key = "observation.state" if feature_type == FeatureType.STATE else "action"
    normalizer = NormalizerProcessorStep(
        features={key: PolicyFeature(type=feature_type, shape=(2,))},
        norm_map={feature_type: mode},
        stats={
            key: {
                "mean": [1.0, 0.0],
                "std": [0.0, 1.0],
                "min": [1.0, 0.0],
                "max": [1.0, 1.0],
                "q01": [1.0, 0.0],
                "q99": [1.0, 1.0],
                "q10": [1.0, 0.0],
                "q90": [1.0, 1.0],
            }
        },
    )
    value = torch.tensor([1.0, 0.5], dtype=dtype)
    transition = (
        {TransitionKey.OBSERVATION: {key: value}}
        if feature_type == FeatureType.STATE
        else {TransitionKey.ACTION: value}
    )
    output = normalizer(transition)
    output_value = (
        output[TransitionKey.OBSERVATION][key]
        if feature_type == FeatureType.STATE
        else output[TransitionKey.ACTION]
    )
    assert output_value.dtype == dtype
    assert bool(torch.isfinite(output_value).all())
    expected_constant = 0.0 if mode == NormalizationMode.MEAN_STD else -1.0
    torch.testing.assert_close(output_value[0], torch.tensor(expected_constant, dtype=dtype), rtol=0, atol=0)
    assert normalizer.eps == 1e-8
    assert normalizer.get_config()["eps"] == 1e-8


@pytest.mark.parametrize("mode", list(NormalizationMode))
def test_processor_preserves_forward_inverse_finiteness_for_fp16(mode):
    if mode not in (
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ):
        pytest.skip("Not a scale normalization mode")
    config = {
        "features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        "norm_map": {FeatureType.ACTION: mode},
        "stats": {
            "action": {"mean": [2.0], "std": [0.0], "min": [2.0], "max": [2.0],
                       "q01": [2.0], "q99": [2.0], "q10": [2.0], "q90": [2.0]},
        },
    }
    for processor_cls in (NormalizerProcessorStep, UnnormalizerProcessorStep):
        processor = processor_cls(**config)
        outcome = processor({TransitionKey.ACTION: torch.tensor([2.0], dtype=torch.float16)})
        assert torch.isfinite(outcome[TransitionKey.ACTION]).all()


@pytest.mark.parametrize("epsilon", [1e-8, 1e-12, 0.0])
def test_fp16_denominator_floor_for_zero_spread(epsilon):
    normalizer = NormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": {"mean": [5.0], "std": [0.0]}},
        eps=epsilon,
    )
    result = normalizer({TransitionKey.ACTION: torch.tensor([5.0], dtype=torch.float16)})
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.zeros(1, dtype=torch.float16), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode",
    [
        NormalizationMode.MEAN_STD,
        NormalizationMode.MIN_MAX,
        NormalizationMode.QUANTILES,
        NormalizationMode.QUANTILE10,
    ],
)
def test_constant_coordinate_backward_has_finite_nonzero_gradient(dtype, mode):
    # Fixing a 0/0 forward NaN is insufficient if backward gradients still
    # diverge. An exactly constant feature is a valid forward training input.
    processor = NormalizerProcessorStep(
        features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(1,))},
        norm_map={FeatureType.STATE: mode},
        stats={
            "observation.state": {
                "mean": [1.0], "std": [0.0], "min": [1.0], "max": [1.0],
                "q01": [1.0], "q99": [1.0], "q10": [1.0], "q90": [1.0],
            }
        },
    )
    obs = torch.tensor([1.0], dtype=dtype, requires_grad=True)
    output = processor({TransitionKey.OBSERVATION: {"observation.state": obs}})
    value = output[TransitionKey.OBSERVATION]["observation.state"]
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert obs.grad is not None
    assert torch.isfinite(obs.grad).all()
    assert bool((obs.grad > 0).all())
