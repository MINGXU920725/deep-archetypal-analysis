# train.py
class Args:
    batch_size = 500
    max_epochs = 50
    latent_dim = 32
    lr = 5e-3
    train_subset_size = 20000
    warmup_epochs = 5
    semantic_epochs = 10
    nu_x_tilde = 300
    gamma = 15.0
    beta = 0.01
    alpha = 0.1
    cos_sim_weight = 1.0
    num_classes = 10

    # dataset = "mnist"     # Change this to select dataset
    dataset = "kmnist"  # Use this to switch to KMNIST

    SEED = None
    num_archetypes_per_class = None
    total_archetypes = None
    exp_dir = None


if __name__ == "__main__":
    from vae_with_aa import train
    import os
    seeds = [46]
    aa_counts = [2,3,4,5,6,7]

    for seed in seeds:
        for aa in aa_counts:

            args = Args()

            # The selected dataset is determined here directly from Args.dataset
            args.SEED = seed
            args.num_archetypes_per_class = aa
            args.total_archetypes = aa * args.num_classes

            # Automatically name experiment folder based on dataset
            exp_dir = f"results/{args.dataset}/seed_{seed}_aa_{aa}"
            os.makedirs(exp_dir, exist_ok=True)

            args.exp_dir = exp_dir

            print(f"\n=============================================")
            print(f"Start experiment: dataset={args.dataset}, seed={seed}, AA-per-class={aa}")
            print(f"Results will be saved to: {exp_dir}")
            print(f"=============================================\n")

            train(args)
