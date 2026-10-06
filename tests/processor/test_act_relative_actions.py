from __future__ import annotations

from collections import deque

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    bind_relative_anchor,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


ACTION_NAMES = ["j0.pos", "j1.pos", "j2.pos", "gripper.pos"]


def _config(*, relative: bool) -> ACTConfig:
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        chunk_size=3,
        n_action_steps=3,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
        use_relative_actions=relative,
        relative_exclude_joints=["gripper"],
        action_feature_names=list(ACTION_NAMES),
    )


def _stats():
    return {
        OBS_STATE: {"mean": torch.zeros(4), "std": torch.ones(4)},
        OBS_ENV_STATE: {"mean": torch.zeros(4), "std": torch.ones(4)},
        ACTION: {"mean": torch.zeros(4), "std": torch.ones(4)},
    }


def _relative_step(pre):
    return next(step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep))


def test_act_default_processor_layout_is_unchanged():
    pre, post = make_act_pre_post_processors(_config(relative=False), _stats())
    assert len(pre.steps) == 4
    assert len(post.steps) == 2
    assert not any(isinstance(step, RelativeActionsProcessorStep) for step in pre.steps)
    assert not any(isinstance(step, AbsoluteActionsProcessorStep) for step in post.steps)


def test_act_relative_training_converts_before_normalization_and_preserves_gripper():
    pre, _ = make_act_pre_post_processors(_config(relative=True), _stats())
    processed = pre(
        {
            OBS_STATE: torch.tensor([10.0, 20.0, 30.0, 99.0]),
            OBS_ENV_STATE: torch.zeros(4),
            ACTION: torch.tensor([10.1, 20.2, 30.3, 0.8]),
        }
    )

    torch.testing.assert_close(
        processed[ACTION],
        torch.tensor([[0.1, 0.2, 0.3, 0.8]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_act_action_queue_holds_chunk_anchor_until_drain():
    cfg = _config(relative=True)
    pre, post = make_act_pre_post_processors(cfg, _stats())
    policy = ACTPolicy(cfg)
    bind_relative_anchor(policy, pre)
    relative = _relative_step(pre)

    pre({OBS_STATE: torch.tensor([10.0, 10.0, 10.0, 5.0]), OBS_ENV_STATE: torch.zeros(4)})
    torch.testing.assert_close(
        relative.get_cached_state(),
        torch.tensor([[10.0, 10.0, 10.0, 5.0]]),
    )

    policy._action_queue = deque(
        [torch.zeros((1, 4)), torch.zeros((1, 4))],
        maxlen=cfg.n_action_steps,
    )
    pre({OBS_STATE: torch.tensor([20.0, 20.0, 20.0, 6.0]), OBS_ENV_STATE: torch.zeros(4)})

    relative_action = torch.tensor([[0.1, 0.2, 0.3, 0.8]])
    absolute = post(relative_action)
    torch.testing.assert_close(
        absolute,
        torch.tensor([[10.1, 10.2, 10.3, 0.8]]),
        atol=1e-6,
        rtol=1e-6,
    )

    policy._action_queue.clear()
    pre({OBS_STATE: torch.tensor([30.0, 30.0, 30.0, 7.0]), OBS_ENV_STATE: torch.zeros(4)})
    reanchored = post(relative_action)
    torch.testing.assert_close(
        reanchored,
        torch.tensor([[30.1, 30.2, 30.3, 0.8]]),
        atol=1e-6,
        rtol=1e-6,
    )
