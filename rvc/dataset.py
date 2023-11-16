from pathlib import Path
from typing import SupportsIndex, TypedDict

import pydub.silence
import scipy.signal
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import Dataset

from rvc.utils import SAMPLE_RATE, numpy_to_pydub, pydub_to_numpy


class RVCSample(TypedDict):
    audio: torch.Tensor
    f0: torch.Tensor
    f0_coarse: torch.Tensor
    hubert_features: torch.Tensor


class RVCDataset(Dataset[RVCSample]):
    def __init__(
        self,
        path: Path,
        cache_path: Path,
        min_silence_ms: int = 500,
        silence_thresh_dbfs: int = -42,
        keep_silence_ms: int = 500,
        seek_step_ms: int = 15,
        min_sample_ms: int = 1500,
        max_sample_ms: int = 3000,
        sample_overlap_ms: int = 300,
    ) -> None:
        super().__init__()

        self.min_silence_ms = min_silence_ms
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.keep_silence_ms = keep_silence_ms
        self.seek_step_ms = seek_step_ms
        self.min_sample_ms = min_sample_ms
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
            audio, sample_rate = torchaudio.load(p)

            b, a = scipy.signal.butter(
                N=5,
                Wn=48,
                btype="high",
                fs=sample_rate,
            )
            resample = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)

            audio = audio.mean(0)  # 1 channel
            audio = scipy.signal.lfilter(b, a, audio).astype("float32")

            segments = pydub.silence.split_on_silence(
                numpy_to_pydub(audio, sample_rate),
                min_silence_len=self.min_silence_ms,
                silence_thresh=self.silence_thresh_dbfs,
                keep_silence=self.keep_silence_ms,
                seek_step=self.seek_step_ms,
            )
            for idx, segment in enumerate(segments):
                end = 0
                while end < len(segment):
                    start = max(0, end - self.sample_overlap_ms)
                    end = min(start + self.max_sample_ms, len(segment))
                    if end - start < self.min_sample_ms:
                        continue

                    sample = resample(
                        torch.tensor(pydub_to_numpy(segment[start:end]).reshape(1, -1))
                    )
                    segment_path = cache_path / f"{p.stem}_{idx}_{start}_{end}.wav"
                    torchaudio.save(
                        segment_path,
                        sample,
                        SAMPLE_RATE,
                        format="wav",
                    )
