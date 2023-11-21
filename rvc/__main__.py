from pathlib import Path
from typing import Annotated

import fairseq.checkpoint_utils
import torch
from typer import Argument, Typer

from rvc.dataset import RVCDataset
from rvc.rmvpe import RMVPE

app = Typer()


@app.command()
@torch.no_grad()
def train(
    data_path: Annotated[Path, Argument()] = Path("env/data"),
    cache_path: Annotated[Path, Argument()] = Path("env/cache"),
    model_path: Annotated[Path, Argument()] = Path("env/models"),
) -> None:
    rmvpe = RMVPE(model_path / "rmvpe.pt").eval()
    (hubert,), saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path / "hubert_base.pt")],
        suffix="",
    )

    dataset = RVCDataset(data_path, cache_path, rmvpe, hubert)
    print(len(dataset))


app()
exit(0)
