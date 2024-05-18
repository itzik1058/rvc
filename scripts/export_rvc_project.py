from pathlib import Path
from typing import Annotated

import numpy
import torch
import torchaudio
from torchaudio.models import wav2vec2_base
from tqdm import tqdm
from typer import Argument, Typer

from rvc.dataset import RVCDataset
from rvc.rmvpe import RMVPE

main = Typer()


@main.command()
def export_rvc_project(
    data_path: Annotated[Path, Argument()] = Path("env/data"),
    cache_path: Annotated[Path, Argument()] = Path("env/cache"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
    output_path: Annotated[Path, Argument()] = Path("env/rvc-project-data"),
) -> None:
    rmvpe = RMVPE(model_path / "rmvpe.pt").eval()
    wav2vec2 = wav2vec2_base()
    wav2vec2.load_state_dict(torch.load(model_path / "wav2vec2.pt"))

    def feature_extractor(waveform: torch.Tensor) -> torch.Tensor:
        features, _ = wav2vec2.extract_features(waveform)
        return features[-1]

    dataset = RVCDataset(
        data_path,
        cache_path,
        pitch_estimator=rmvpe,
        feature_extractor=feature_extractor,
    )

    (output_path / "0_gt_wavs").mkdir(parents=True, exist_ok=True)
    (output_path / "2a_f0").mkdir(parents=True, exist_ok=True)
    (output_path / "2b-f0nsf").mkdir(parents=True, exist_ok=True)
    (output_path / "3_feature768").mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(dataset)):
        torchaudio.save(
            output_path / "0_gt_wavs" / f"{i}.wav",
            sample.audio,
            48_000,
            format="wav",
            encoding="PCM_F",
        )
        torch.save(sample.spectrogram, output_path / "0_gt_wavs" / f"{i}.spec.pt")
        numpy.save(output_path / "2a_f0" / f"{i}.wav.npy", sample.f0_coarse)
        numpy.save(output_path / "2b-f0nsf" / f"{i}.wav.npy", sample.f0)
        numpy.save(output_path / "3_feature768" / f"{i}.npy", sample.features)
