from pathlib import Path
from typing import Annotated

import numpy
import torch
import torchaudio
from tqdm import tqdm
from typer import Argument, Option, Typer

from rvc.model.rmvpe import RMVPE
from rvc.pipeline.dataset import RVCDataset
from rvc.pipeline.preprocess import make_feature_extractor

main = Typer()


@main.command()
def export_rvc_project(
    data_path: Annotated[Path, Argument()],
    model_path: Annotated[Path, Argument()],
    output_path: Annotated[Path, Argument()],
    cache_path: Annotated[Path | None, Option()] = None,
) -> None:
    rmvpe = RMVPE(model_path / "rmvpe.pt").eval()
    feature_extractor = make_feature_extractor(model_path / "wav2vec2.pt")

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


if __name__ == "__main__":
    main()
