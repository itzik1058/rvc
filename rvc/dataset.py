import logging
import math
from pathlib import Path
from typing import Callable, SupportsIndex, TypedDict

import pydub.silence
import scipy.signal
import torch
import torchaudio
import torchaudio.transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from rvc.utils import SAMPLE_RATE, TARGET_SAMPLE_RATE, numpy_to_pydub, pydub_to_numpy

F0_BIN = 256
F0_MAX = 1100.0
F0_MIN = 50.0
F0_MEL_MIN = 1127 * math.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * math.log(1 + F0_MAX / 700)


class RVCSample(TypedDict):
    audio: torch.Tensor
    f0: torch.Tensor
    f0_coarse: torch.Tensor
    features: torch.Tensor


class RVCDataset(Dataset[RVCSample]):
    def __init__(
        self,
        path: Path,
        cache_path: Path,
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

        self.pitch_estimator = pitch_estimator
        self.feature_extractor = feature_extractor
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
        path = self.samples[index]
        audio, _ = torchaudio.load(path.with_suffix(".wav"))
        return {
            "audio": audio,
            "spectrogram": torch.load(path.with_suffix(".spec")),
            "f0": torch.load(path.with_suffix(".f0")),
            "f0_coarse": torch.load(path.with_suffix(".f0c")),
            "features": torch.load(path.with_suffix(".ft")),
        }

    def __len__(self) -> int:
        return len(self.samples)

    @torch.inference_mode()
    def _load(self, path: Path, cache_path: Path) -> None:
        for p in tqdm(list(path.iterdir())):
            if not p.is_file():
                continue
            try:
                audio, sample_rate = torchaudio.load(p)
            except Exception:
                logging.error(f"failed to load {p.name}")
                continue

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

                    segment_path = cache_path / f"{p.stem}_{idx}_{start}_{end}"
                    segment_lock = segment_path.with_suffix(".lock")
                    if segment_lock.is_file():
                        self.samples.append(segment_path)
                        continue

                    sample = torch.tensor(
                        pydub_to_numpy(segment[start:end]).reshape(1, -1)
                    )
                    # smooth with normalized sample
                    alpha = 0.75
                    sample = (
                        sample / sample.abs().max() * 0.9 * alpha + (1 - alpha) * sample
                    )

                    torchaudio.save(
                        segment_path.with_suffix(".wav"),
                        sample,
                        TARGET_SAMPLE_RATE,
                        format="wav",
                        encoding="PCM_F",
                    )

                    spectrogram = torchaudio.transforms.Spectrogram(
                        n_fft=2048,
                        win_length=2048,
                        hop_length=TARGET_SAMPLE_RATE // 100,
                        power=1,
                    )
                    torch.save(
                        spectrogram(sample).squeeze(0),
                        segment_path.with_suffix(".spec"),
                    )

                    sample_feature = resample_feature(sample.float())
                    f0 = self.pitch_estimator(sample_feature)
                    torch.save(f0, segment_path.with_suffix(".f0"))

                    f0_mel = 1127 * torch.log(1 + f0 / 700)
                    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * (
                        F0_BIN - 2
                    ) / (F0_MEL_MAX - F0_MEL_MIN) + 1
                    f0_mel[f0_mel <= 1] = 1
                    f0_mel[f0_mel > F0_BIN - 1] = F0_BIN - 1
                    f0_coarse = f0_mel.round().int()
                    torch.save(
                        f0_coarse,
                        segment_path.with_suffix(".f0c"),
                    )

                    features = self.feature_extractor(sample_feature)
                    torch.save(
                        features.squeeze(0),
                        segment_path.with_suffix(".ft"),
                    )

                    segment_lock.touch(exist_ok=True)

                    self.samples.append(segment_path)
