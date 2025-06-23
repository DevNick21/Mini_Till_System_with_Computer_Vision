# train_embedding.py
import torch
import random
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from PIL import Image
import os

# 1) Build your triplet dataset


class HandwritingTripletDataset(Dataset):
    def __init__(self, root, transform):
        # root/writer_0/*.jpg, root/writer_1/*.jpg, ...
        self.paths = []
        self.labels = []
        self.transform = transform
        for label, writer in enumerate(sorted(os.listdir(root))):
            folder = os.path.join(root, writer)
            for fn in os.listdir(folder):
                self.paths.append(os.path.join(folder, fn))
                self.labels.append(label)
        # group paths by label
        self.by_label = {}
        for p, l in zip(self.paths, self.labels):
            self.by_label.setdefault(l, []).append(p)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        anc_path = self.paths[idx]
        anc_label = self.labels[idx]
        # positive: different file, same label
        pos_path = anc_path
        while pos_path == anc_path:
            pos_path = random.choice(self.by_label[anc_label])
        # negative: any sample from a different label
        neg_label = random.choice([l for l in self.by_label if l != anc_label])
        neg_path = random.choice(self.by_label[neg_label])

        def img(p): return self.transform(Image.open(p).convert("RGB"))
        return img(anc_path), img(pos_path), img(neg_path)

# 2) Lightning module


class TripletNet(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.embedder = nn.Sequential(*list(backbone.children())[:-1])
        self.loss_fn = nn.TripletMarginLoss(margin=1.0)
        self.lr = lr

    def forward(self, x):
        return self.embedder(x).flatten(1)

    def training_step(self, batch, _):
        a, p, n = batch
        ea, ep, en = self(a), self(p), self(n)
        loss = self.loss_fn(ea, ep, en)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],  # mean per channel
                    [0.5, 0.5, 0.5])  # std per channel
    ])
    ds = HandwritingTripletDataset("slips", transform)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    model = TripletNet()
    Trainer(max_epochs=10).fit(model, dl)
    # save the fine-tuned weights
    torch.save(model.embedder.state_dict(), "cv_service/writer_embedder.pt")
