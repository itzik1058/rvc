import math
from pathlib import Path

import faiss
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from rvc.pipeline.dataset import RVCDataset


@torch.inference_mode()
def export_faiss_index(
    dataset: RVCDataset,
    checkpoint_path: Path,
) -> None:
    features = []
    for sample in tqdm(dataset, "faiss index"):
        features.append(sample.features)
    all_features = torch.concat(features)

    if all_features.size(0) >= 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=10000,
            # verbose=True,
            batch_size=256,
            compute_labels=False,
            init="random",
        )
        all_features = kmeans.fit(all_features).cluster_centers_

    n_ivf = min(
        int(16 * math.sqrt(all_features.shape[0])),
        all_features.shape[0] // 39,
    )
    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(all_features)
    batch_size_add = 8192
    for i in range(0, all_features.shape[0], batch_size_add):
        index.add(all_features[i : i + batch_size_add])
    faiss.write_index(
        index,
        checkpoint_path.with_suffix(".index").as_posix(),
    )
