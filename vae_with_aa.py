import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import numpy as np
from sklearn.cluster import KMeans
import os
import json
import pickle

# ----------------------------
# Main training function (receives external args and saves all results)
# ----------------------------
def train(args):
    # Create experiment directory
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    # Logging function: print to console and also save to file
    log_file = os.path.join(exp_dir, "train.log")
    def print_log(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # Initialize random seeds
    def set_seed(s=args.SEED):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"[Init] Using device: {device}")

    # Load parameters from args
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    latent_dim = args.latent_dim
    lr = args.lr
    train_subset_size = args.train_subset_size
    warmup_epochs = args.warmup_epochs
    semantic_epochs = args.semantic_epochs
    nu_x_tilde = args.nu_x_tilde
    gamma = args.gamma
    beta = args.beta
    num_archetypes_per_class = args.num_archetypes_per_class
    alpha = args.alpha
    cos_sim_weight = args.cos_sim_weight
    total_archetypes = args.total_archetypes
    img_channels, img_size, num_classes = 1, 28, 10

    # Save experiment parameters
    params_path = os.path.join(exp_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print_log(f"[Config] Parameters saved to: {params_path}")

    # ----------------------------
    # Data loading
    # ----------------------------
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "mnist":
        full_train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    elif args.dataset == "kmnist":
        full_train_set = datasets.KMNIST("./data", train=True, download=True, transform=transform)
        test_set = datasets.KMNIST("./data", train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader_full = DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Prepare full data grouped by class
    def prepare_class_full_data(full_train_set, device):
        class_full_imgs = {}
        for img, label in full_train_set:
            label = int(label)
            if label not in class_full_imgs:
                class_full_imgs[label] = []
            class_full_imgs[label].append(img.unsqueeze(0))
        for label in class_full_imgs:
            class_full_imgs[label] = torch.cat(class_full_imgs[label]).to(device)
        return class_full_imgs

    class_full_imgs = prepare_class_full_data(full_train_set, device)

    # ----------------------------
    # Model definition (AA initialization modified)
    # ----------------------------
    class VAEWithAA(torch.nn.Module):
        def __init__(self, latent_dim=32, img_channels=1, img_size=28, num_classes=10):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
                torch.nn.Flatten(),
                torch.nn.Linear(7*7*64, 512), torch.nn.ReLU(),
                torch.nn.Linear(512, latent_dim * 2)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 512), torch.nn.ReLU(),
                torch.nn.Linear(512, 7*7*64),
                torch.nn.Unflatten(1, (64, 7, 7)),
                torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 1, 4, 2, 1), torch.nn.Sigmoid()
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 128), torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes)
            )
            self.arch_logits = torch.nn.ParameterDict()
            self.class_full_mu = {}
            self.fc_sample_to_aa = torch.nn.Linear(latent_dim, total_archetypes)

        def encode(self, x):
            h = self.encoder(x)
            mu, logvar = torch.chunk(h, 2, dim=1)
            mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        # Improved AA initialization
        def init_class_archetypes(self, class_full_imgs, device):
            print_log("[Init] Initializing class-wise archetypes (multi-seed + fallback)...")
            for label, imgs in class_full_imgs.items():
                mu, _ = self.encode(imgs)
                self.class_full_mu[label] = mu.detach()
                mu_np = mu.detach().cpu().numpy()
                latent_dim = mu.size(1)
                target_num = num_archetypes_per_class

                # If latent collapse: expand with small noise
                if np.std(mu_np) < 1e-3:
                    mu_np += 0.05 * np.random.randn(*mu_np.shape)

                # Multi-seed KMeans
                centers_np = None
                actual_clusters = 0
                for seed_offset in range(5):
                    try_seed = args.SEED + seed_offset
                    km = KMeans(
                        n_clusters=target_num,
                        random_state=try_seed,
                        n_init=10,
                        max_iter=300
                    ).fit(mu_np)
                    actual_clusters = len(np.unique(km.labels_))
                    if actual_clusters >= max(2, target_num // 2):
                        centers_np = km.cluster_centers_
                        break

                # If KMeans fails: fallback to random selection
                if centers_np is None:
                    print_log(f"[Warning] KMeans failed for class {label}. Using random initialization.")
                    centers_np = mu_np[np.random.choice(len(mu_np), target_num, replace=False)]

                centers = torch.tensor(centers_np, device=device, dtype=mu.dtype)
                centers += 0.05 * torch.randn_like(centers)

                # Remove duplicates
                dist_matrix = torch.cdist(centers, centers)
                for i in range(target_num):
                    for j in range(i + 1, target_num):
                        if dist_matrix[i, j] < 1e-6:
                            centers[j] += 0.1 * torch.randn_like(centers[j])

                # Initialize logits
                dists = torch.cdist(centers, mu)
                idx = dists.topk(k=20, largest=False).indices
                logits = torch.zeros(target_num, mu.size(0), device=device)
                for i in range(target_num):
                    logits[i, idx[i]] = 5.0 + 0.5 * torch.randn_like(logits[i, idx[i]])

                self.arch_logits[str(label)] = torch.nn.Parameter(logits, requires_grad=True)

            print_log("[Init] Archetypes initialized.")

        def get_all_archetypes(self):
            all_arch = []
            for label in sorted(self.arch_logits.keys()):
                W = F.softmax(self.arch_logits[label], dim=1)
                X_mu_full = self.class_full_mu[int(label)]
                all_arch.append(W @ X_mu_full)
            return torch.cat(all_arch, dim=0)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            sample_to_aa_weights = F.softmax(self.fc_sample_to_aa(mu), dim=1)
            x_hat = self.decode(z)
            return x_hat, z, mu, logvar, torch.zeros(1).to(device), sample_to_aa_weights

    # (remaining functions unchanged, only comments/logs translated)
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def recon_loss(x_hat, x):
        return F.mse_loss(x_hat, x, reduction='mean')

    def estimate_mutual_info_ty(mu, classifier, num_classes=10):
        logits = classifier(mu)
        probs = F.softmax(logits, dim=1)
        entropy_cond = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        return -entropy_cond

    def cos_similarity_loss(mu_batch, all_archetypes, weights_batch):
        recon_sample = torch.matmul(weights_batch, all_archetypes)
        cos_sim = F.cosine_similarity(recon_sample, mu_batch, dim=1).mean()
        return (1 - cos_sim)

    def total_aa_loss(model, mu, y):
        all_archetypes = model.get_all_archetypes()
        weights_batch = F.softmax(model.fc_sample_to_aa(mu), dim=1)
        loss_sample = F.mse_loss(torch.matmul(weights_batch, all_archetypes), mu)
        loss_cos = cos_similarity_loss(mu, all_archetypes, weights_batch)
        total_loss = loss_sample + cos_sim_weight * loss_cos
        return total_loss, loss_sample.item(), loss_cos.item()

    # ----------------------------
    # Training loop (unchanged)
    # ----------------------------
    def compute_epoch_losses(model, loader, device, is_train, optimizer=None, epoch=0):
        model.train() if is_train else model.eval()
        total_total = total_samples = 0.0
        use_aa_beta = (epoch >= semantic_epochs)

        with torch.set_grad_enabled(is_train):
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                x_hat, z, mu, logvar, _, _ = model(x)
                I_tx = kl_loss(mu, logvar)
                recon = recon_loss(x_hat, x)
                I_ty_raw = estimate_mutual_info_ty(mu, model.classifier, num_classes)
                I_ty_display = -I_ty_raw

                if use_aa_beta:
                    L_at, avg_Ls, avg_La = total_aa_loss(model, mu, y)
                else:
                    L_at, avg_Ls, avg_La = torch.tensor(0.0).to(device), 0.0, 0.0

                if epoch < warmup_epochs:
                    total_loss = I_tx + nu_x_tilde * recon
                elif epoch < semantic_epochs:
                    total_loss = I_tx - (beta * 0.5) * I_ty_raw + nu_x_tilde * recon
                else:
                    total_loss = I_tx - beta * I_ty_raw + nu_x_tilde * recon + gamma * L_at

                if is_train:
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                bs = x.size(0)
                total_total += total_loss.item() * bs
                total_samples += bs

                if (batch_idx + 1) % 10 == 0:
                    print_log(f"[Train] Batch {batch_idx+1:03d} | I(t;x): {I_tx.item():.4f} | "
                              f"I(t;y): {I_ty_display.item():.4f} | Recon: {recon.item():.4f} | "
                              f"Total: {total_loss.item():.4f}")
                    if use_aa_beta:
                        print_log(f"   ↳ AA sample loss: {avg_Ls:.6f} | AA cos loss: {avg_La:.6f}")

        return total_total / total_samples

    def compute_class_archetype_distance(model):
        avg_distances = {}
        for label in model.arch_logits.keys():
            W = F.softmax(model.arch_logits[label], dim=1)
            X_mu_full = model.class_full_mu[int(label)]
            archetypes = W @ X_mu_full
            if num_archetypes_per_class >= 2:
                d1 = torch.norm(archetypes[0] - archetypes[1], p=2).item()
            else:
                d1 = 0.0
            avg_distances[int(label)] = d1
        return sum(avg_distances.values()) / len(avg_distances)

    model = VAEWithAA(latent_dim=latent_dim, img_channels=img_channels, 
                      img_size=img_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": lr}])

    epoch_losses = []
    print_log(f"[Training Start] {max_epochs} epochs total")
    print_log(f"Warmup: first {warmup_epochs} epochs | Semantic: {warmup_epochs+1}–{semantic_epochs} | AA starts from {semantic_epochs+1}")
    print_log(f"[Config] batch_size={batch_size} | lr={lr} | gamma={gamma} | cos_sim_weight={cos_sim_weight}\n")

    for epoch in range(max_epochs):
        use_aa_beta = (epoch >= semantic_epochs)
        print_log(f"===================== Epoch {epoch+1}/{max_epochs} =====================")
        if epoch < warmup_epochs:
            print_log("Mode: Warmup (KL + Reconstruction)")
        elif epoch < semantic_epochs:
            print_log("Mode: Semantic (KL + Recon + weak MI)")
        else:
            print_log("Mode: Full (KL + Recon + MI + full AA reconstruction + cos loss)")

        rand_idx = random.sample(range(len(full_train_set)), train_subset_size)
        train_subset = Subset(full_train_set, rand_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)

        if epoch == semantic_epochs:
            model.init_class_archetypes(class_full_imgs, device)
            for p in model.arch_logits.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": model.arch_logits.parameters(), "lr": 1e-2})

        train_total_loss = compute_epoch_losses(model, train_loader, device, True, optimizer, epoch)
        epoch_losses.append(train_total_loss)
        print_log(f"\n[Summary] Epoch {epoch+1} Train Total Loss: {train_total_loss:.4f}")

        if use_aa_beta:
            avg_dist = compute_class_archetype_distance(model)
            print_log(f"[Collapse Monitor] Avg intra-class archetype distance: {avg_dist:.4f}")
        print_log("=======================================================================\n")

    loss_path = os.path.join(exp_dir, "epoch_total_losses.pkl")
    with open(loss_path, "wb") as f:
        pickle.dump(epoch_losses, f)
    print_log(f"[Save] Epoch losses saved to: {loss_path}")

    if max_epochs > semantic_epochs and hasattr(model, 'get_all_archetypes'):
        try:
            all_archetypes = model.get_all_archetypes().cpu().detach().numpy()
            aa_path = os.path.join(exp_dir, "final_archetypes.npy")
            np.save(aa_path, all_archetypes)
            print_log(f"[Save] Final archetype coordinates saved to: {aa_path} (shape: {all_archetypes.shape})")

            decoded_archetypes = model.decode(torch.tensor(all_archetypes, device=device)).cpu().detach().numpy()
            decoded_path = os.path.join(exp_dir, "final_archetypes_decoded.npy")
            np.save(decoded_path, decoded_archetypes)
            print_log(f"[Save] Decoded archetype images saved to: {decoded_path} (shape: {decoded_archetypes.shape})")
            
        except Exception as e:
            print_log(f"[Warning] Failed to save archetypes: {str(e)}")

    aa_weight_path = os.path.join(exp_dir, "fc_sample_to_aa_weights.pth")
    torch.save(model.fc_sample_to_aa.state_dict(), aa_weight_path)
    print_log(f"[Save] Sample-to-AA weight parameters saved to: {aa_weight_path}")

    model_path = os.path.join(exp_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print_log(f"[Save] Final model weights saved to: {model_path}")


# # =========================================
# # ✅ 简易导出：只定义结构，不依赖训练闭包
# # =========================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class VAEWithAA(nn.Module):
#     """结构与训练时一致，用于加载 model_final.pth"""
#     def __init__(self, latent_dim=32, img_channels=1, img_size=28, total_archetypes=50, num_classes=10):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(img_channels, 32, 3, stride=2, padding=1), nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.Flatten(),
#             nn.Linear((img_size // 4) * (img_size // 4) * 64, 512), nn.ReLU(),
#             nn.Linear(512, latent_dim * 2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512), nn.ReLU(),
#             nn.Linear(512, (img_size // 4) * (img_size // 4) * 64),
#             nn.Unflatten(1, (64, img_size // 4, img_size // 4)),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(32, img_channels, 4, 2, 1), nn.Sigmoid()
#         )
#         self.fc_sample_to_aa = nn.Linear(latent_dim, total_archetypes)


#     def encode(self, x):
#         h = self.encoder(x)
#         mu, logvar = torch.chunk(h, 2, dim=1)
#         mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-8)
#         return mu, logvar

#     def decode(self, z):
#         return self.decoder(z)
    
#     def forward(self, x):
#         """标准推理流程"""
#         mu, _ = self.encode(x)
#         return self.decode(mu)
    
#     def get_all_archetypes(self, exp_dir="results/mnist/seed_42_aa_5"):
#         """
#         从保存目录加载 archetype 坐标，用于 AA 重构
#         """
#         aa_path = os.path.join(exp_dir, "final_archetypes.npy")
#         if not os.path.exists(aa_path):
#             raise FileNotFoundError(f"❌ 未找到 archetype 文件: {aa_path}")
#         all_arch = np.load(aa_path)
#         return torch.tensor(all_arch, dtype=torch.float32)