import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pydub.silence
import scipy.signal
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import IterableDataset

from rvc.utils import SAMPLE_RATE, TARGET_SAMPLE_RATE, numpy_to_pydub, pydub_to_numpy

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


@dataclass(frozen=True)
class RVCSample:
    audio: torch.Tensor
    spectrogram: torch.Tensor
    f0: torch.Tensor
    f0_coarse: torch.Tensor
    features: torch.Tensor


class RVCDataset(IterableDataset[RVCSample]):
    def __init__(
        self,
        path: Path,
        cache_path: Path | None,
        pitch_estimator: Callable[[torch.Tensor], torch.Tensor],
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        min_silence_ms: int = 500,
        silence_thresh_dbfs: int = -42,
        keep_silence_ms: int = 500,
        seek_step_ms: int = 15,
        min_sample_ms: int = 1500,
        max_sample_ms: int = 3000,
        sample_overlap_ms: int = 300,
    ) -> None:
        super().__init__()

        self.path = path
        self.cache_path = cache_path

        self.pitch_estimator = pitch_estimator
        self.feature_extractor = feature_extractor
        self.min_silence_ms = min_silence_ms
        self.silence_thresh_dbfs = silence_thresh_dbfs
        self.keep_silence_ms = keep_silence_ms
        self.seek_step_ms = seek_step_ms
        self.min_sample_ms = min_sample_ms
        self.max_sample_ms = max_sample_ms
        self.sample_overlap_ms = sample_overlap_ms

    def __iter__(self) -> Iterable[RVCSample]:
        for p in self.path.iterdir():
            if not p.is_file():
                continue
            if self.cache_path is not None:
                cache = self.cache_path / f"{p.name}"
                cache.mkdir(parents=True, exist_ok=True)
                cache_lock = cache / "lock"
                if cache_lock.is_file():
                    yield from self._load(cache)
                    continue
            try:
                audio, sample_rate = torchaudio.load(p)
            except Exception:
                logging.error(f"failed to load {p.name}")
                continue
            for i, sample in enumerate(self._preprocess(audio, sample_rate)):
                yield sample
                if self.cache_path is not None:
                    segment_path = cache / f"{i}"
                    self._save_sample(sample, segment_path)
            if self.cache_path is not None:
                cache_lock.touch(exist_ok=True)

    @torch.inference_mode()
    def _preprocess(self, audio: torch.Tensor, sample_rate: int) -> Iterable[RVCSample]:
        b, a = scipy.signal.butter(
            N=5,
            Wn=48,
            btype="high",
            fs=sample_rate,
        )
        resample_target = torchaudio.transforms.Resample(
            sample_rate, TARGET_SAMPLE_RATE
        )
        resample_feature = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)

        audio = resample_target(audio)
        audio = audio.mean(0)  # 1 channel
        audio = scipy.signal.lfilter(b, a, audio)

        segments = pydub.silence.split_on_silence(
            numpy_to_pydub(audio, TARGET_SAMPLE_RATE),
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

                sample = torch.tensor(pydub_to_numpy(segment[start:end]).reshape(1, -1))
                # smooth with normalized sample
                alpha = 0.75
                sample = (
                    sample / sample.abs().max() * 0.9 * alpha + (1 - alpha) * sample
                )

                spectrogram = torchaudio.transforms.Spectrogram(
                    n_fft=2048,
                    win_length=2048,
                    hop_length=TARGET_SAMPLE_RATE // 100,
                    power=1,
                )

                sample_feature = resample_feature(sample.float())
                f0 = self.pitch_estimator(sample_feature)

                f0_mel = 1127 * torch.log(1 + f0 / 700)
                f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * (
                    F0_BIN - 2
                ) / (F0_MEL_MAX - F0_MEL_MIN) + 1
                f0_mel[f0_mel <= 1] = 1
                f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
                f0_coarse = f0_mel.round().int()

                features = self.feature_extractor(sample_feature)

                yield RVCSample(
                    audio=sample,
                    spectrogram=spectrogram(sample).squeeze(0),
                    f0=f0,
                    f0_coarse=f0_coarse,
                    features=features.squeeze(0),
                )

    def _load(self, path: Path) -> Iterable[RVCSample]:
        for p in path.glob("*.wav"):
            audio, _ = torchaudio.load(p.with_suffix(".wav"))
            yield RVCSample(
                audio=audio,
                spectrogram=torch.load(p.with_suffix(".spec")),
                f0=torch.load(p.with_suffix(".f0")),
                f0_coarse=torch.load(p.with_suffix(".f0c")),
                features=torch.load(p.with_suffix(".ft")),
            )

    def _save_sample(self, sample: RVCSample, path: Path):
        torchaudio.save(
            path.with_suffix(".wav"),
            sample.audio,
            TARGET_SAMPLE_RATE,
            format="wav",
            encoding="PCM_F",
        )
        torch.save(
            sample.spectrogram,
            path.with_suffix(".spec"),
        )
        torch.save(sample.f0, path.with_suffix(".f0"))
        torch.save(
            sample.f0_coarse,
            path.with_suffix(".f0c"),
        )
        torch.save(
            sample.features,
            path.with_suffix(".ft"),
        )
