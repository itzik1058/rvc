import logging
from pathlib import Path
from typing import Annotated

import torch
from torch.utils.data import DataLoader
from torchaudio.models import wav2vec2_base
from typer import Argument, Typer

from rvc.dataset import RVCDataset
from rvc.rmvpe import RMVPE

logging.basicConfig()

app = Typer()


@app.command()
def train(
    data_path: Annotated[Path, Argument()] = Path("env/data"),
    cache_path: Annotated[Path, Argument()] = Path("env/cache"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
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

    data_loader = DataLoader(dataset)
