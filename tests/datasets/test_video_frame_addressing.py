from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.video_utils import (
    FrameTimestampError,
    decode_video_frames_torchcodec,
)


class _FakeFrames:
    def __init__(self, pts):
        self.data = torch.zeros((len(pts), 3, 2, 2), dtype=torch.uint8)
        self.pts_seconds = torch.tensor(pts, dtype=torch.float64)


class _FakeDecoder:
    def __init__(self, *, average_fps=29.97, pts=(16.7,)):
        self.metadata = SimpleNamespace(average_fps=average_fps)
        self._pts = pts
        self.requested = None

    def get_frames_at(self, *, indices):
        self.requested = list(indices)
        return _FakeFrames(self._pts)


class _FakeCache:
    def __init__(self, decoder):
        self.decoder = decoder

    def get_decoder(self, video_path):
        return self.decoder


def test_explicit_logical_index_bypasses_decoder_average_fps():
    decoder = _FakeDecoder(average_fps=29.97, pts=(16.7,))
    out = decode_video_frames_torchcodec(
        "unused.mp4",
        [16.7],
        1e-6,
        decoder_cache=_FakeCache(decoder),
        return_uint8=True,
        frame_indices=[501],
    )

    assert decoder.requested == [501]
    assert out.shape == (1, 3, 2, 2)


def test_legacy_timestamp_path_reconstructs_different_index():
    decoder = _FakeDecoder(average_fps=29.97, pts=(16.7,))
    decode_video_frames_torchcodec(
        "unused.mp4",
        [16.7],
        1e-6,
        decoder_cache=_FakeCache(decoder),
        return_uint8=True,
    )

    assert decoder.requested == [500]


def test_index_addressing_still_validates_pts():
    decoder = _FakeDecoder(average_fps=29.97, pts=(16.6,))

    with pytest.raises(FrameTimestampError, match="tolerance"):
        decode_video_frames_torchcodec(
            "unused.mp4",
            [16.7],
            1e-3,
            decoder_cache=_FakeCache(decoder),
            return_uint8=True,
            frame_indices=[501],
        )


def _reader():
    reader = object.__new__(DatasetReader)
    reader._meta = SimpleNamespace(
        total_episodes=4,
        video_keys=["cam"],
        episodes=[
            {
                "length": 3,
                "dataset_from_index": 0,
                "videos/cam/chunk_index": 0,
                "videos/cam/file_index": 0,
            },
            {
                "length": 5,
                "dataset_from_index": 3,
                "videos/cam/chunk_index": 0,
                "videos/cam/file_index": 0,
            },
            {
                "length": 2,
                "dataset_from_index": 8,
                "videos/cam/chunk_index": 0,
                "videos/cam/file_index": 1,
            },
            {
                "length": 4,
                "dataset_from_index": 10,
                "videos/cam/chunk_index": 0,
                "videos/cam/file_index": 0,
            },
        ],
    )
    reader._video_file_frame_offsets = None
    return reader


def test_reader_derives_file_local_offsets_without_schema_change():
    reader = _reader()
    offsets = reader._build_video_file_frame_offsets()

    assert offsets[(0, "cam")] == 0
    assert offsets[(1, "cam")] == 3
    assert offsets[(2, "cam")] == 0
    assert offsets[(3, "cam")] == 8


def test_reader_maps_delta_window_to_file_local_indices():
    reader = _reader()
    got = reader._get_query_video_frame_indices(
        abs_idx=4,
        ep_idx=1,
        query_indices={"cam": [3, 4, 7]},
    )

    assert got["cam"] == [3, 4, 7]


def test_reader_current_frame_on_file_rollover_starts_from_zero():
    reader = _reader()
    got = reader._get_query_video_frame_indices(
        abs_idx=8,
        ep_idx=2,
        query_indices=None,
    )

    assert got["cam"] == [0]


def test_frame_indices_require_non_negative_integers():
    decoder = _FakeDecoder(pts=(0.0,))
    with pytest.raises(ValueError, match="non-negative integers"):
        decode_video_frames_torchcodec(
            "unused.mp4",
            [0.0],
            1e-6,
            decoder_cache=_FakeCache(decoder),
            return_uint8=True,
            frame_indices=[True],
        )
