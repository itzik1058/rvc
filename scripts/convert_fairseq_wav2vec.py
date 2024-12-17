# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "fairseq",
#     "typer-slim",
# ]
# ///

from pathlib import Path

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from typer import Typer

main = Typer()


@main.command(
    help="https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md"
)
def convert_fairseq_wav2vec(fairseq_path: Path, output_path: Path):
    (model,), _, _ = load_model_ensemble_and_task(
        [fairseq_path.as_posix()],
        suffix="",
    )
    wav2vec2 = import_fairseq_model(model)
    torch.save(wav2vec2.state_dict(), output_path)


if __name__ == "__main__":
    main()
