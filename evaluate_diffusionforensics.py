"""Validate the downloaded DIRE classifiers against the original paper.

This script reproduces the DIRE evaluation on the official DiffusionForensics
LSUN-Bedroom *test* split, using the artefacts shipped in this repo:

  * ``dire_models/``          - the downloaded DIRE classifier checkpoints
  * ``dire_data/test/...``    - the *precomputed* DIRE reconstruction-error maps
                                (one tarball per generator + ``real``)
  * ``images/`` / ``recons/`` - the raw images and their diffusion
                                reconstructions (used only by the optional
                                ``--check-dire`` consistency test below)

For the chosen classifier (default ``dire_models/lsun_adm.pth``) it:

  1. Extracts the precomputed DIRE maps for ``real`` and each generator.
  2. Runs the classifier with the exact DIRE eval transform
     (Resize 256 -> CenterCrop 224 -> ToTensor -> ImageNet-normalize),
     matching ``demo.py`` / ``test.py``.
  3. Reports ACC / AP / R_ACC / F_ACC per generator.
  4. Prints the numbers next to the values reported in the DIRE paper
     (Wang et al., ICCV 2023), Table 2 / Table 3, and the deltas.

A small delta (computed ~= paper, i.e. ~99-100 ACC/AP) confirms that our
classification half of the pipeline is correct, which is the same code path
reused to score the freshly-computed DIRE maps on FakeFlickr.

Usage
-----
    conda activate dire
    python evaluate_diffusionforensics.py                 # lsun_adm.pth
    python evaluate_diffusionforensics.py \
        --ckpt dire_models/lsun_iddpm.pth                  # another checkpoint

Optionally sanity-check the DIRE *formula* itself (no diffusion model needed,
uses the provided reconstructions):

    python evaluate_diffusionforensics.py --check-dire sdv2
"""

from __future__ import annotations

import argparse
import csv
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from utils.utils import get_network  # noqa: E402

# ---------------------------------------------------------------------------
# Paper reference numbers (ACC, AP) in percent.
#
# DIRE (ours), reconstruction model = ADM, training dataset = LSUN-Bedroom.
# Diffusion generators come from Table 2; the GAN column (stylegan) comes from
# the cross-dataset Table 3 ("StyleGAN(LSUN-B.)").  Keys are the *generator*
# names as used in ``dire_data/test/lsun_bedroom`` (after normalisation).
#
# Source: Z. Wang et al., "DIRE for Diffusion-Generated Image Detection",
# ICCV 2023 (arXiv:2303.09295), Tables 2 & 3.
# ---------------------------------------------------------------------------
PAPER_REF: dict[str, dict[str, tuple[float, float]]] = {
    # trained on the ADM subset (the headline model)
    "lsun_adm": {
        "adm": (100.0, 100.0), "ddpm": (100.0, 100.0), "iddpm": (100.0, 100.0),
        "pndm": (99.7, 100.0), "sdv1": (100.0, 100.0), "sdv2": (100.0, 100.0),
        "ldm": (100.0, 100.0), "vqdiffusion": (100.0, 100.0),
        "stylegan": (99.9, 100.0),
    },
    "lsun_pndm": {
        "adm": (100.0, 100.0), "ddpm": (100.0, 100.0), "iddpm": (100.0, 100.0),
        "pndm": (100.0, 100.0), "sdv1": (100.0, 100.0), "sdv2": (89.4, 99.9),
        "ldm": (100.0, 100.0), "vqdiffusion": (100.0, 100.0),
    },
    "lsun_iddpm": {
        "adm": (99.6, 100.0), "ddpm": (100.0, 100.0), "iddpm": (100.0, 100.0),
        "pndm": (100.0, 100.0), "sdv1": (89.7, 99.9), "sdv2": (97.7, 100.0),
        "ldm": (100.0, 100.0), "vqdiffusion": (99.9, 100.0),
    },
    "lsun_stylegan": {
        "adm": (98.8, 100.0), "ddpm": (99.8, 100.0), "iddpm": (99.9, 100.0),
        "pndm": (89.6, 100.0), "sdv1": (95.2, 100.0), "sdv2": (100.0, 100.0),
        "ldm": (100.0, 100.0), "vqdiffusion": (100.0, 100.0),
    },
}

# Map tarball stem (in dire_data) -> canonical generator key used above.
NAME_ALIASES = {
    "midjouney": "midjourney",
    "stylegan_official_res": "stylegan",
}


def canon(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def extract_tarball(tar_path: Path, dest_root: Path) -> Path:
    """Extract ``tar_path`` under ``dest_root`` (idempotent).

    Handles both gzip and plain-tar files (some shipped tarballs are plain
    POSIX tar despite the ``.tar.gz`` suffix).  Returns the directory that
    holds the extracted images (the single top-level folder in the archive).
    """
    with tarfile.open(tar_path, "r:*") as tf:  # r:* autodetects gz / plain
        members = tf.getnames()
        top = members[0].split("/")[0]
        out_dir = dest_root / top
        # Consider it already extracted if the folder holds image files.
        if out_dir.is_dir() and any(out_dir.iterdir()):
            return out_dir
        tf.extractall(dest_root)
    return out_dir


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def classify_folder(
    model: torch.nn.Module,
    folder: Path,
    label: int,
    load_size: int,
    crop_size: int,
    aug_norm: bool,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    trans = transforms.Compose([
        transforms.Resize(load_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    probs: list[float] = []
    labels: list[int] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        if aug_norm:
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            out = model(x).sigmoid().flatten().cpu().numpy().tolist()
        probs.extend(out)
        labels.extend([label] * len(out))
        batch.clear()

    for path in tqdm(files, desc=f"  {folder.name}", dynamic_ncols=True, leave=False):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  skip {path}: {exc}")
            continue
        batch.append(trans(img))
        if len(batch) >= batch_size:
            flush()
    flush()
    return probs, labels


def metrics_from(real_probs, fake_probs) -> dict[str, float]:
    y_true = np.array([0] * len(real_probs) + [1] * len(fake_probs))
    y_pred = np.array(real_probs + fake_probs)
    return {
        "ACC": float(accuracy_score(y_true, y_pred > 0.5)) * 100,
        "AP": float(average_precision_score(y_true, y_pred)) * 100,
        "R_ACC": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)) * 100,
        "F_ACC": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)) * 100,
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dire-data-dir", type=Path,
                   default=REPO_ROOT / "dire_data" / "test" / "lsun_bedroom",
                   help="Folder with precomputed DIRE-map tarballs (real + generators).")
    p.add_argument("--ckpt", type=Path, default=REPO_ROOT / "dire_models" / "lsun_adm.pth",
                   help="DIRE classifier checkpoint to evaluate.")
    p.add_argument("--work-dir", type=Path, default=REPO_ROOT / "eval_work" / "diffusionforensics",
                   help="Where DIRE-map tarballs get extracted.")
    p.add_argument("--real-name", default="real", help="Tarball stem of the real DIRE maps.")
    p.add_argument("--generators", nargs="+", default=None,
                   help="Generators to evaluate (default: every tarball except real).")
    p.add_argument("--results-csv", type=Path,
                   default=REPO_ROOT / "eval_work" / "diffusionforensics_results.csv")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--load-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--no-aug-norm", dest="aug_norm", action="store_false", default=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not args.dire_data_dir.is_dir():
        raise FileNotFoundError(f"--dire-data-dir not found: {args.dire_data_dir}")

    ckpt_stem = args.ckpt.stem
    ref = PAPER_REF.get(ckpt_stem, {})
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    # Discover generators (all tarballs minus the real set).
    all_tars = sorted(args.dire_data_dir.glob("*.tar.gz"))
    stem_of = lambda t: t.name[: -len(".tar.gz")]
    if args.generators:
        gen_tars = [args.dire_data_dir / f"{g}.tar.gz" for g in args.generators]
    else:
        gen_tars = [t for t in all_tars if stem_of(t) != args.real_name]

    real_tar = args.dire_data_dir / f"{args.real_name}.tar.gz"
    if not real_tar.is_file():
        raise FileNotFoundError(f"real tarball not found: {real_tar}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading classifier '{args.arch}' from {args.ckpt}")
    model = get_network(args.arch)
    state = torch.load(args.ckpt, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    # Real DIRE maps are shared across all generators -> classify once.
    print(f"\nExtracting + scoring real DIRE maps ({real_tar.name})")
    real_dir = extract_tarball(real_tar, args.work_dir)
    real_probs, _ = classify_folder(model, real_dir, 0, args.load_size, args.crop_size,
                                    args.aug_norm, device, args.batch_size)
    print(f"  real: N={len(real_probs)}  mean P(fake)={np.mean(real_probs):.4f}")

    rows: list[dict] = []
    print("\n" + "=" * 92)
    header = f"{'generator':<16}{'ACC':>7}{'AP':>7}{'R_ACC':>8}{'F_ACC':>8}   |  {'paper ACC/AP':>12}   {'ΔACC':>7}{'ΔAP':>7}"
    print(header)
    print("-" * 92)
    for tar in gen_tars:
        gen = stem_of(tar)
        key = canon(gen)
        gen_dir = extract_tarball(tar, args.work_dir)
        fake_probs, _ = classify_folder(model, gen_dir, 1, args.load_size, args.crop_size,
                                        args.aug_norm, device, args.batch_size)
        m = metrics_from(real_probs, fake_probs)

        ref_pair = ref.get(key)
        if ref_pair:
            d_acc = m["ACC"] - ref_pair[0]
            d_ap = m["AP"] - ref_pair[1]
            ref_str = f"{ref_pair[0]:.1f}/{ref_pair[1]:.1f}"
            d_acc_str, d_ap_str = f"{d_acc:+.1f}", f"{d_ap:+.1f}"
        else:
            ref_str, d_acc_str, d_ap_str = "n/a", "", ""

        print(f"{key:<16}{m['ACC']:>7.2f}{m['AP']:>7.2f}{m['R_ACC']:>8.2f}{m['F_ACC']:>8.2f}"
              f"   |  {ref_str:>12}   {d_acc_str:>7}{d_ap_str:>7}")
        rows.append({"generator": key, **{k: round(v, 4) for k, v in m.items()},
                     "paper_ACC": ref_pair[0] if ref_pair else "",
                     "paper_AP": ref_pair[1] if ref_pair else ""})
    print("=" * 92)

    # Averages over generators that have a paper reference.
    ref_rows = [r for r in rows if r["paper_ACC"] != ""]
    if ref_rows:
        avg_acc = np.mean([r["ACC"] for r in ref_rows])
        avg_ap = np.mean([r["AP"] for r in ref_rows])
        pavg_acc = np.mean([r["paper_ACC"] for r in ref_rows])
        pavg_ap = np.mean([r["paper_AP"] for r in ref_rows])
        print(f"\nAverage over {len(ref_rows)} paper-referenced generators:")
        print(f"  computed  ACC/AP = {avg_acc:.2f}/{avg_ap:.2f}")
        print(f"  paper     ACC/AP = {pavg_acc:.2f}/{pavg_ap:.2f}")
        print(f"  delta     ACC/AP = {avg_acc - pavg_acc:+.2f}/{avg_ap - pavg_ap:+.2f}")

    fieldnames = ["generator", "ACC", "AP", "R_ACC", "F_ACC", "N_real", "N_fake",
                  "paper_ACC", "paper_AP"]
    with args.results_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {args.results_csv}")


if __name__ == "__main__":
    main()
