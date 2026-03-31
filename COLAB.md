# Google Colab

Use [colab/hermes_google_colab.ipynb](/home/shakilahmed/work/code/golf/colab/hermes_google_colab.ipynb) to run HERMES on a Colab GPU while keeping `data/`, `checkpoints/`, and `runs/` on Google Drive.

The notebook does four things:

1. Mounts Google Drive.
2. Clones this repo from GitHub into `/content/golf`.
3. Symlinks Drive-backed storage into the repo so checkpoints survive runtime resets.
4. Runs the same training command you used locally, including `--resume latest`.

Recommended Colab path layout:

```text
MyDrive/golf-colab/
  data/
  checkpoints/
  runs/
```

If you already have local checkpoints you want in Colab, upload them into:

```text
MyDrive/golf-colab/checkpoints/hermes_v1_fineweb_ode8/
```

Then the notebook's default training cell will resume from the latest checkpoint for that run.

If you need to regenerate FineWeb slices inside Colab, the notebook includes an optional prep cell that runs:

```bash
python prepare_fineweb.py --subset CC-MAIN-2024-10 --format text --train-mb 64 --val-mb 8
```

Notes:

- Colab already includes PyTorch, so the notebook only installs the dataset-related Python packages.
- The notebook clones `https://github.com/ahmedshakill/golf.git` directly.
- Training code stays unchanged; Colab uses the existing `--train_path`, `--val_path`, `--checkpoint_dir`, and `--resume` flags.
- Keep the repo on `/content` for speed and store artifacts on Drive for persistence.
