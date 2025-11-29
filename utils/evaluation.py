import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import os


def evaluate_reconstruction_from_saved(exp_dir="./results/mnist_notebook_seed42_aa5", dataset="mnist"):
    """
    Function: Compute the MSE between the original images and their direct
            reconstructions / AA reconstructions using the saved decoded files
            (supports both MNIST and KMNIST).

    Arguments:
        - exp_dir: directory containing the saved results
        - dataset: dataset type ("mnist" or "kmnist")
    """
    # -----------------------------

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        ds = datasets.MNIST("./data", train=True, transform=transform, download=True)
    elif dataset == "kmnist":
        ds = datasets.KMNIST("./data", train=True, transform=transform, download=True)
    else:
        raise ValueError("dataset only supports 'mnist' or 'kmnist'")

    # -----------------------------

    X = np.stack([img.numpy() for img, _ in ds])  # (N,1,28,28)
    y = np.array([label for _, label in ds])      

    # -----------------------------

    direct_recon = np.load(os.path.join(exp_dir, "decoded_mu.npy"))   
    aa_recon = np.load(os.path.join(exp_dir, "decoded_mu_AA.npy"))     

    # -----------------------------

    mse_direct = np.mean((direct_recon - X) ** 2, axis=(1,2,3))  
    mse_aa = np.mean((aa_recon - X) ** 2, axis=(1,2,3))          

    # -----------------------------
    results = {
        int(c): {
            "Direct": mse_direct[y == c].mean(),
            "AA": mse_aa[y == c].mean()
        }
        for c in np.unique(y)
    }
    results["ALL"] = {
        "Direct": mse_direct.mean(),
        "AA": mse_aa.mean()
    }

    # -----------------------------
    print(f"\n==== {dataset.upper()} Reconstruction MSE ====")
    for c in sorted(k for k in results.keys() if isinstance(k, int)):
        print(f"Class {c}: Direct={results[c]['Direct']:.6f} | AA={results[c]['AA']:.6f}")
    print(f"Overall: Direct={results['ALL']['Direct']:.6f} | AA={results['ALL']['AA']:.6f}")
    print("=============================================")

