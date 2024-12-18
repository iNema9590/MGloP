import pathlib
# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from sklearn import linear_model
from torch import nn
from torch.utils import data
from tqdm import tqdm, trange


# Example-specific constants
BASE_DIR = pathlib.Path('data')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOGFISH_RAW_PATH = BASE_DIR /"dataset_dog-fish_train-900_test-300.npz"
DOGFISH_EMB_PATH = BASE_DIR /"dataset_coda.npz"

DOGFISH_RAW_MMAP = np.load(str(DOGFISH_RAW_PATH), mmap_mode="r")

L2_WEIGHT = 1e-4


def load_dogfish_examples(split, idxs):
    image = DOGFISH_RAW_MMAP[f"X_{split}"][idxs]
    label = DOGFISH_RAW_MMAP[f"Y_{split}"][idxs]
    return image, label


def load_inception_embeds():
    if DOGFISH_EMB_PATH.is_file():
        print(f"Using cached Inceptionv3 embeddings: {DOGFISH_EMB_PATH}")
        return np.load(str(DOGFISH_EMB_PATH))

    # load pretrained Inceptionv3
    model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # remove the last layer
    model.eval()
    model = model.to(DEVICE)

    embeds = {}

    for split in ("train", "test"):
        n = 2 * (900 if (split == "train") else 300)

        X, Y = [], []
        for idx in trange(0, n, 20, desc=f"Embedding Dogfish ({split})"):
            batch_idxs = [idx + offset for offset in range(20)]
            images, labels = load_dogfish_examples(split, batch_idxs)

            # getting Inceptionv3 embedding
            with torch.no_grad():
                images = torch.tensor(images, dtype=torch.float32, device=DEVICE)
                images = images.permute([0, 3, 1, 2])
                h = model(images).cpu().numpy()

            X.append(h)
            Y.append(labels)

        embeds[f"X_{split}"] = np.concatenate(X, axis=0).astype(np.float32)
        embeds[f"Y_{split}"] = np.concatenate(Y, axis=0).astype(np.float32)

    np.savez(str(DOGFISH_EMB_PATH), **embeds)
    return embeds


def fit_model(X, Y):
    C = 1 / (X.shape[0] * L2_WEIGHT)
    sk_clf = linear_model.LogisticRegression(C=C, tol=1e-8, max_iter=1000)
    sk_clf = sk_clf.fit(X.numpy(), Y.numpy())

    # recreate model in PyTorch
    fc = nn.Linear(2048, 1, bias=True)
    fc.weight = nn.Parameter(torch.tensor(sk_clf.coef_))
    fc.bias = nn.Parameter(torch.tensor(sk_clf.intercept_))

    pt_clf = nn.Sequential(
        fc,
        nn.Flatten(start_dim=-2),
        nn.Sigmoid()
    )

    pt_clf = pt_clf.to(device=DEVICE, dtype=torch.float32)
    return pt_clf

def captioned_image(model, embeds, split, idx):
    x = embeds[f"X_{split}"][idx]
    y = embeds[f"Y_{split}"][idx].item()

    x = torch.tensor(x, device=DEVICE).unsqueeze(0)
    y_hat = model(x).round().item()

    # turn image into [0 .. 255] RGB image
    image, _ = load_dogfish_examples(split, idx)
    image = (image * 127.5) + 127.5
    image = image.clip(min=0.0, max=255.0)
    image = np.round(image).astype(np.uint8)

    return image

   
    






    
    
