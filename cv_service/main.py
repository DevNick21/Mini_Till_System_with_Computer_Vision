from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import numpy as np
import cv2
import hdbscan
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
import faiss    # <— new dependency

app = FastAPI()

# Load the pre-trained embedding model
_backbone = resnet18(weights=None)
_embed_model = nn.Sequential(*list(_backbone.children())[:-1])

# — drop weights_only, just load normally —
state = torch.load("writer_embedder.pt",
                   map_location="cpu", weights_only=True)
_embed_model.load_state_dict(state, strict=False)

_embed_model.eval()
_device = torch.device("cpu")
_embed_model.to(_device)

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


def get_embedding(img: np.ndarray) -> np.ndarray:
    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = _embed_model(tensor)
    return emb.squeeze().cpu().numpy()


@app.post("/cluster")
async def cluster(files: List[UploadFile] = File(...)) -> List[dict]:
    ids, embs = [], []
    for f in files:
        data = await f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(400, f"Bad image: {f.filename}")
        embs.append(get_embedding(img))
        try:
            ids.append(int(f.filename.split(".", 1)[0]))
        except:
            raise HTTPException(
                400, f"Filename must start with numeric id: {f.filename}")

    n = len(ids)
    if n == 0:
        raise HTTPException(400, "No files submitted")

    X = np.vstack(embs).astype(np.float32)  # shape (n,512)
    EXPECTED = 6

    # 1) HDBSCAN density-based clustering
    hdb = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1)
    core_labels = hdb.fit_predict(X)  # -1 = noise
    real_clusters = [c for c in set(core_labels) if c >= 0]

    # 2) Decide whether to use HDBSCAN or FAISS K-means
    if len(real_clusters) < EXPECTED:
        # FAISS KMeans fallback
        d = X.shape[1]
        kmeans = faiss.Kmeans(d, EXPECTED, niter=20, verbose=False)
        kmeans.train(X)
        centroids = kmeans.centroids           # shape (EXPECTED, d)
        # assign each point to nearest centroid
        _, labels = kmeans.index.search(X, 1)  # returns (distances, labels)
        labels = labels.flatten().tolist()
    else:
        # Keep HDBSCAN and assign noise by nearest centroid
        centroids = {c: X[core_labels == c].mean(
            axis=0) for c in real_clusters}
        labels = core_labels.copy().tolist()
        for i, c in enumerate(core_labels):
            if c == -1:
                # find closest cluster centroid
                closest = min(centroids, key=lambda k: np.linalg.norm(
                    X[i] - centroids[k]))
                labels[i] = int(closest)

    # 3) Return mapping
    return [{"id": ids[i], "cluster": int(labels[i])} for i in range(n)]
