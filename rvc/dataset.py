from pathlib import Path
from typing import SupportsIndex, TypedDict

import pydub
import pydub.silence
import scipy.signal
from torch import Tensor
from torch.utils.data import Dataset

from rvc.utils import load_audio


class RVCSample(TypedDict):
    audio: Tensor
    f0: Tensor
    f0_coarse: Tensor
    hubert_features: Tensor


class RVCDataset(Dataset[RVCSample]):
    def __init__(
        self,
        path: Path,
        cache_path: Path,
        source_sampling_rate_hz: int = 40_000,
        target_sampling_rate_hz: int = 16_000,
        min_silence_ms: int = 400,
        silence_thresh_dbfs: int = -42,
        keep_silence_ms: int = 500,
        seek_step_ms: int = 15,
        max_sample_ms: int = 3000,
        sample_overlap_ms: int = 300,
    ) -> None:
        super().__init__()

        self.source_sampling_rate_hz = source_sampling_rate_hz
        self.target_sampling_rate_hz = target_sampling_rate_hz
        self.min_silence_ms = min_silence_ms
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.keep_silence_ms = keep_silence_ms
        self.seek_step_ms = seek_step_ms
        self.max_sample_ms = max_sample_ms
        self.sample_overlap_ms = sample_overlap_ms

        self.samples: list[Path] = []
        self._load(path, cache_path)

    def __getitem__(self, index: SupportsIndex) -> RVCSample:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path, cache_path: Path) -> None:
        for p in path.iterdir():
            audio = load_audio(p, self.source_sampling_rate_hz)
            b, a = scipy.signal.butter(
                N=5, Wn=48, btype="high", fs=self.source_sampling_rate_hz
            )
            audio = scipy.signal.lfilter(b, a, audio)
            segments = pydub.silence.split_on_silence(
                pydub.AudioSegment(audio),
                min_silence_len=self.min_silence_ms,
                silence_thresh=self.silence_thresh_dbfs,
                keep_silence=self.keep_silence_ms,
                seek_step=self.seek_step_ms,
            )
            for segment in segments:
                end = 0
                while end < len(segment):
                    start = max(0, end - self.sample_overlap_ms)
                    end = min(start + self.max_sample_ms, len(segment))
                    sample = segment[start:end]
                    sample.set_frame_rate(self.target_sampling_rate_hz)
                    sample.export(cache_path / f"{p.name}_{end}.wav", format="wav")
