"""Independent PR #4862 online inference contract regression (expected RED).

PR author owns the implementation. This review reproducer is intentionally
isolated in the reviewer's fork, NOT a competing upstream PR. The shipped
DiffusionPolicy.reset() omits subclass-provided extra observation queues,
while predict_action_chunk() discards all keys not in self._queues once
state/history queues are populated. Thus a model that CAN train on new
conditioning inputs cannot infer online with the inherited queue policy.
"""
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, DiffusionModel
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

EXTRA="observation.extra"
E=5


class ExtraObsDiffusion(DiffusionModel):
    def _extra_global_cond_dim(self, config: DiffusionConfig)->int:
        return E

    def _extra_global_cond_feats(self, batch):
        # Same shape contract as original PR's test WiderModel.
        return batch[EXTRA]


class ExtraObsPolicy(DiffusionPolicy):
    def _make_diffusion_model(self, config: DiffusionConfig):
        return ExtraObsDiffusion(config)

    def _extra_observation_queue_keys(self) -> tuple[str, ...]:
        # Explicit sensor history contract; only keys used by the new
        # diffusion conditioning branch are added, not arbitrary metadata.
        return (EXTRA,)


def _config():
    cfg=DiffusionConfig(
        horizon=8,n_action_steps=4,n_obs_steps=2,down_dims=(64,128)
    )
    cfg.input_features={
        OBS_STATE:PolicyFeature(FeatureType.STATE,(6,)),
        OBS_ENV_STATE:PolicyFeature(FeatureType.ENV,(4,))
    }
    cfg.output_features={ACTION:PolicyFeature(FeatureType.ACTION,(6,))}
    cfg.normalization_mapping=dict.fromkeys(
        ("STATE","ACTION","VISUAL","ENV"),
        NormalizationMode.MEAN_STD
    )
    return cfg


def test_extra_conditioning_trains_but_offline_and_online_remain_equivalent():
    torch.manual_seed(401)
    model=ExtraObsPolicy(_config())
    # Training: already tested in upstream, and should legitimately work.
    train={
        OBS_STATE:torch.randn(1,2,6),
        OBS_ENV_STATE:torch.randn(1,2,4),
        EXTRA:torch.randn(1,2,E),
        ACTION:torch.randn(1,8,6),
        "action_is_pad":torch.zeros(1,8,dtype=torch.bool)
    }
    with torch.no_grad():
        loss,_=model(train)
    assert bool(torch.isfinite(loss))
    # A genuine ONLINE observation is one tick, not a pre-stacked
    # sequence. The existing inherited queue must retain every
    # declared extra input, otherwise _extra_global_cond_feats raises
    # KeyError after predict_action_chunk filters the live batch.
    obs={
        OBS_STATE:torch.randn(1,6),
        OBS_ENV_STATE:torch.randn(1,4),
        EXTRA:torch.randn(1,E)
    }
    assert EXTRA in obs
    # This is the red test. Upstream does not create EXTRA in reset
    # and populate_queues ignores that input; the first chunk loses it.
    action=model.select_action(obs,noise=torch.zeros(1,8,6))
    assert action.shape==(1,6)


def test_parent_queue_contract_exposes_missing_subclass_input():
    model=ExtraObsPolicy(_config())
    assert EXTRA in model._queues, (
        "Online queued inference MUST retain the subclass's extra "
        "conditioning feature; PR #4862 reset() only creates built-in keys."
    )


def test_extra_observation_history_maintains_tick_order_in_online_action_queue():
    """Online feature provenance must distinguish previous/current observation."""
    torch.manual_seed(944)
    policy=ExtraObsPolicy(_config())
    def obs(value):
        return {
            OBS_STATE:torch.zeros(1,6),
            OBS_ENV_STATE:torch.zeros(1,4),
            EXTRA:torch.full((1,E),float(value)),
        }
    # First observation is repeated until the n_obs_steps buffer is full.
    first=policy.select_action(obs(1),noise=torch.zeros(1,8,6))
    assert first.shape==(1,6)
    assert len(policy._queues[EXTRA])==2
    assert all(torch.equal(x,torch.ones(1,E)) for x in policy._queues[EXTRA])

    # Even with remaining actions in its action queue, a new actual
    # observation must advance the extra feature history.
    second=policy.select_action(obs(2))
    assert second.shape==(1,6)
    assert torch.equal(policy._queues[EXTRA][0],torch.ones(1,E))
    assert torch.equal(policy._queues[EXTRA][1],torch.full((1,E),2.))


def test_conflicting_or_non_observation_queue_declared_names_are_refused():
    class DuplicateQueuePolicy(ExtraObsPolicy):
        def _extra_observation_queue_keys(self):
            return (OBS_STATE,)
    class InvalidQueuePolicy(ExtraObsPolicy):
        def _extra_observation_queue_keys(self):
            return ("control.hidden_state",)

    import pytest
    with pytest.raises(ValueError,match="Invalid/duplicate"):
        DuplicateQueuePolicy(_config())
    with pytest.raises(ValueError,match="Invalid/duplicate"):
        InvalidQueuePolicy(_config())


def test_missing_new_observation_refuses_online_inference_not_silent_reuse():
    import pytest
    policy=ExtraObsPolicy(_config())
    obs={OBS_STATE:torch.zeros(1,6),OBS_ENV_STATE:torch.zeros(1,4)}
    with pytest.raises(KeyError,match="observation.extra"):
        policy.select_action(obs,noise=torch.zeros(1,8,6))
