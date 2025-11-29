
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# -----------------------------
def plot_decoded_archetypes(exp_dir, n_cols=10):
    """
    Visualize the saved archetype decoded images.
    Relies only on final_archetypes_decoded.npy and does not load any model.
    """
    decoded_path = os.path.join(exp_dir, "final_archetypes_decoded.npy")
    if not os.path.exists(decoded_path):
        raise FileNotFoundError(f" File not found: {decoded_path}")

    decoded = np.load(decoded_path)
    num_imgs = decoded.shape[0]
    n_rows = int(np.ceil(num_imgs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < num_imgs:
            ax.imshow(decoded[i, 0], cmap="gray")
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    # plt.suptitle("VAE Archetypes (Decoded Images)", fontsize=14, y=1.02)
    plt.show()

# -----------------------------
# Function 2: visualize_latent_archetypes_3d (VAE-specific 3D visualization)
# -----------------------------
def visualize_latent_archetypes_3d(exp_dir, dataset="mnist", latent_dim=32, n_samples=2000, seed=42):
    """Visualize the latent space and the archetype convex hull (independent of vae_with_aa)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    model_path = os.path.join(exp_dir, "model_final.pth")
    arch_path = os.path.join(exp_dir, "final_archetypes.npy")
    if not os.path.exists(model_path) or not os.path.exists(arch_path):
        raise FileNotFoundError("can not find model_final.pth or final_archetypes.npy")

    class EncoderOnly(nn.Module):
        def __init__(self, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
                nn.Linear(512, latent_dim * 2)
            )
        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = torch.chunk(h, 2, dim=1)
            mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
            return mu

    ckpt = torch.load(model_path, map_location=device)
    encoder = EncoderOnly(latent_dim).to(device)
    encoder_dict = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
    encoder.encoder.load_state_dict(encoder_dict, strict=False)
    encoder.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        ds = datasets.MNIST("./data", train=True, transform=transform, download=True)
    elif dataset == "kmnist":
        ds = datasets.KMNIST("./data", train=True, transform=transform, download=True)
    else:
        raise ValueError("dataset only supports 'mnist' or 'kmnist'")

    idx = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    subset = torch.stack([ds[i][0] for i in idx]).to(device)

    with torch.no_grad():
        mu = encoder(subset).cpu().numpy()

    A_global = np.load(arch_path)
    Z_joint = np.vstack([mu, A_global])
    tsne = TSNE(n_components=3, init="pca", random_state=seed, perplexity=30)
    Z_emb = tsne.fit_transform(Z_joint)
    Z_emb_data = Z_emb[:len(mu)]
    Z_emb_aa = Z_emb[len(mu):]

    hull = ConvexHull(Z_emb_aa)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Z_emb_data[:, 0], Z_emb_data[:, 1], Z_emb_data[:, 2],
               s=5, alpha=0.2, color='gray', label='Samples')
    ax.scatter(Z_emb_aa[:, 0], Z_emb_aa[:, 1], Z_emb_aa[:, 2],
               marker='^', s=60, color='red', edgecolors='k', linewidths=0.5, label='AA Points')
    for simplex in hull.simplices:
        pts = Z_emb_aa[simplex]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'r-', linewidth=0.8, alpha=0.7)

    # ax.set_title("3D t-SNE of Latent Space and Archetype Convex Hull", fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout()
    plt.show()

    
def visualize_ae_latent_3d(exp_dir, n_samples=2000, seed=42):
    """
    3D visualization for AE latent space + archetypes (VAE-style)
    Uses only saved AE results:
        - latent_mu.npy
        - final_archetypes.npy
    Displays:
        - gray sample points (latent representations)
        - red archetype triangles
        - convex hull connecting archetypes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial import ConvexHull
    import os

    latent_path = os.path.join(exp_dir, "latent_mu.npy")
    arche_path = os.path.join(exp_dir, "final_archetypes.npy")

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"latent_mu.npy not found in: {latent_path}")
    if not os.path.exists(arche_path):
        raise FileNotFoundError(f"final_archetypes.npy not found in: {arche_path}")

    latent = np.load(latent_path)
    archetypes = np.load(arche_path)

    np.random.seed(seed)
    sample_idx = np.random.choice(len(latent), min(n_samples, len(latent)), replace=False)
    sample_latent = latent[sample_idx]

    print(f"[Step 1/3] Reducing {latent.shape[1]}D latent space to 3D using t-SNE...")
    combined_data = np.vstack([sample_latent, archetypes])
    tsne = TSNE(
        n_components=3,
        init="pca",
        random_state=seed,
        perplexity=30,
        n_iter=1000
    )
    combined_3d = tsne.fit_transform(combined_data)

    sample_3d = combined_3d[:len(sample_latent)]
    arche_3d = combined_3d[len(sample_latent):]

    print(f"[Step 2/3] Computing convex hull for {archetypes.shape[0]} archetypes...")
    try:
        hull = ConvexHull(arche_3d)
    except:
        print("[Warning] Archetypes are too close to form a valid convex hull.")
        hull = None

    print("[Step 3/3] Plotting 3D latent space visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Gray points: latent samples
    ax.scatter(
        sample_3d[:, 0], sample_3d[:, 1], sample_3d[:, 2],
        s=6, alpha=0.25, color="gray", label="Samples"
    )

    # Red triangles: archetypes
    ax.scatter(
        arche_3d[:, 0], arche_3d[:, 1], arche_3d[:, 2],
        marker="^", s=80, color="red", edgecolor="black", linewidth=0.8,
        label=f"Archetypes ({archetypes.shape[0]})"
    )

    # Red convex hull edges
    if hull is not None:
        for simplex in hull.simplices:
            pts = arche_3d[simplex]
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2],
                "r-", linewidth=0.8, alpha=0.7
            )

    # Figure styling
    # ax.set_title("AE Latent Space + Archetype Convex Hull (3D t-SNE)", fontsize=13, pad=15)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout()
    plt.show()
