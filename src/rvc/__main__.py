import logging
from pathlib import Path
from typing import Annotated

import torch
from typer import Argument, Option, Typer

from rvc.pipeline import train_pipeline, voice_conversion

logging.basicConfig()

app = Typer()


@app.command()
def train(
    checkpoint_path: Annotated[Path, Argument()] = Path("env/rvc.pt"),
    data_path: Annotated[Path, Argument()] = Path("env/data"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
    cache_path: Annotated[Path | None, Option()] = None,
    device: Annotated[str, Option()] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_pipeline(checkpoint_path, data_path, model_path, cache_path, device)


@app.command()
def inference(
    input_path: Annotated[Path, Argument()],
    checkpoint_path: Annotated[Path, Argument()] = Path("env/rvc.pt"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
) -> None:
    voice_conversion(input_path, checkpoint_path, model_path)
