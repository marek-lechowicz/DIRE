"""Test DIRE on the FakeFlickr dataset using the Flickr30k test split.

For every generator under ``<dataset_root>/generated/<gen>/img`` this script:

  1. Filters images to the IDs listed in the Flickr30k test split.
  2. Stages them into an ``ImageFolder`` layout (``0_real``/``1_fake``)
     under ``<stage_root>/<gen>``. Real images come from
     ``<dataset_root>/real`` (or ``real_rescaled`` for the
     ``flux_fill_real_rescaled`` generator, which was conditioned on
     the rescaled reals).
  3. Computes DIRE reconstruction-error maps with the unconditional
     guided-diffusion model (``guided-diffusion/compute_dire.py``).
  4. Loads the trained DIRE classifier (``--ckpt``) and reports
     ACC / AP / R_ACC / F_ACC over the DIRE maps.

Results are written as one CSV row per generator.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
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

DEFAULT_GENERATORS = [
    "sd_1_5",
    "sd_3_5_large",
    "sdxl_turbo",
    "z_image_turbo",
    "flux_1_dev",
    "flux_fill_flux_1_dev",
    "flux_fill_sd_3_5_large",
    "flux_fill_real_rescaled",
]

# Generators that were conditioned on the rescaled-real source images.
# For these, the matching "real" is the rescaled PNG, not the original JPG.
RESCALED_REAL_GENS = {"flux_fill_real_rescaled"}

# Diffusion-model flags from guided-diffusion/compute_dire.sh
# (256x256 unconditional ImageNet checkpoint).
DIFFUSION_MODEL_FLAGS = [
    "--attention_resolutions", "32,16,8",
    "--class_cond", "False",
    "--diffusion_steps", "1000",
    "--dropout", "0.1",
    "--image_size", "256",
    "--learn_sigma", "True",
    "--noise_schedule", "linear",
    "--num_channels", "256",
    "--num_head_channels", "64",
    "--num_res_blocks", "2",
    "--resblock_updown", "True",
    "--use_fp16", "True",
    "--use_scale_shift_norm", "True",
]
DIFFUSION_SAMPLE_FLAGS = [
    "--timestep_respacing", "ddim20",
    "--use_ddim", "True",
]


def read_test_ids(split_file: Path) -> list[str]:
    with split_file.open("r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Test split is empty: {split_file}")
    return ids


def find_image(dirpath: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPEG"):
        cand = dirpath / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def stage_files(
    stage_parent: Path,
    src_dir: Path,
    test_ids: list[str],
    subfolder: str,
) -> int:
    """Symlink the test-split images from ``src_dir`` into

        ``stage_parent/<subfolder>/``

    (a single-class ImageFolder-style layout, so compute_dire_single's
    ``--has_subfolder True`` emits maps under ``dire_dir/<subfolder>/``).
    Returns the number of images staged.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    dest = stage_parent / subfolder
    if stage_parent.exists():
        shutil.rmtree(stage_parent)
    dest.mkdir(parents=True)

    n = 0
    for img_id in test_ids:
        img = find_image(src_dir, img_id)
        if img is None:
            continue
        os.symlink(img, dest / img.name)
        n += 1
    if n == 0:
        raise RuntimeError(f"No test-split images matched under {src_dir}")
    return n


def run_compute_dire(
    images_dir: Path,
    recons_dir: Path,
    dire_dir: Path,
    diffusion_ckpt: Path,
    num_samples: int,
    batch_size: int,
    cuda_visible_devices: str,
    jpeg_equalize: bool,
    crop_mode: str,
    output_ext: str,
) -> None:
    """Materialise DIRE maps via the MPI-free single-process script.

    This machine has a single GPU and no MPI runtime, so we call
    ``compute_dire_single.py`` (a drop-in for the original MPI-based
    ``compute_dire.py``) instead of ``mpiexec ... compute_dire.py``.
    """

    # The subprocess runs with cwd=guided-diffusion, so all paths must be
    # absolute (the staged dirs live under --work-dir relative to the repo).
    cmd = [
        sys.executable, "compute_dire_single.py",
        "--model_path", str(Path(diffusion_ckpt).resolve()),
        "--images_dir", str(Path(images_dir).resolve()),
        "--recons_dir", str(Path(recons_dir).resolve()),
        "--dire_dir", str(Path(dire_dir).resolve()),
        "--batch_size", str(batch_size),
        "--num_samples", str(num_samples),
        "--has_subfolder", "True",
        "--jpeg_equalize", str(jpeg_equalize),
        "--crop_mode", crop_mode,
    ]
    if output_ext:
        cmd.extend(["--output_ext", output_ext])
    cmd.extend(DIFFUSION_MODEL_FLAGS)
    cmd.extend(DIFFUSION_SAMPLE_FLAGS)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    print(f"[compute_dire] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT / "guided-diffusion"), env=env, check=True)


def classify_folder(
    model: torch.nn.Module,
    folder: Path,
    label: int,
    cfg_load_size: int,
    cfg_crop_size: int,
    aug_norm: bool,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    trans = transforms.Compose([
        transforms.Resize(cfg_load_size),
        transforms.CenterCrop(cfg_crop_size),
        transforms.ToTensor(),
    ])

    probs: list[float] = []
    labels: list[int] = []
    batch: list[torch.Tensor] = []

    def flush(batch_tensors: list[torch.Tensor]) -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        if aug_norm:
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            out = model(x).sigmoid().flatten().cpu().numpy().tolist()
        probs.extend(out)
        labels.extend([label] * len(out))

    for path in tqdm(files, desc=f"  classify {folder.parent.name}/{folder.name}",
                     dynamic_ncols=True, leave=False):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # corrupt / unreadable
            print(f"  skip {path}: {exc}")
            continue
        batch.append(trans(img))
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)
    return probs, labels


def evaluate_generator(
    model: torch.nn.Module,
    real_dire_dir: Path,
    fake_dire_dir: Path,
    cfg_load_size: int,
    cfg_crop_size: int,
    aug_norm: bool,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    real_probs, real_labels = classify_folder(
        model, real_dire_dir, 0, cfg_load_size, cfg_crop_size, aug_norm, device, batch_size,
    )
    fake_probs, fake_labels = classify_folder(
        model, fake_dire_dir, 1, cfg_load_size, cfg_crop_size, aug_norm, device, batch_size,
    )

    y_true = np.array(real_labels + fake_labels)
    y_pred = np.array(real_probs + fake_probs)

    return {
        "ACC": float(accuracy_score(y_true, y_pred > 0.5)),
        "AP": float(average_precision_score(y_true, y_pred)),
        "R_ACC": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)),
        "F_ACC": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)),
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/fake-flickr"),
                   help="Root of the fake-flickr dataset.")
    p.add_argument("--test-split", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"),
                   help="File with one Flickr30k image ID per line (the test split).")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to the trained DIRE classifier checkpoint (.pth).")
    p.add_argument("--diffusion-ckpt", type=Path, required=True,
                   help="Path to the unconditional guided-diffusion checkpoint "
                        "(e.g. 256x256_diffusion_uncond.pt).")
    p.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS,
                   help=f"Generator subdirs to evaluate (default: {DEFAULT_GENERATORS}).")
    p.add_argument("--work-dir", type=Path,
                   default=REPO_ROOT / "data" / "fake_flickr",
                   help="Where to stage symlinks, reconstructions, and DIRE maps.")
    p.add_argument("--results-csv", type=Path,
                   default=REPO_ROOT / "data" / "results" / "fake_flickr_dire.csv",
                   help="Output CSV path.")
    p.add_argument("--arch", default="resnet50",
                   help="Classifier architecture (must match the checkpoint).")
    p.add_argument("--load-size", type=int, default=256)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--aug-norm", action="store_true", default=True)
    p.add_argument("--crop-mode", choices=["resize", "crop"], default="resize",
                   help="How to reach 256px for DIRE: 'resize' (short-side resize + crop, "
                        "guided-diffusion default) or 'crop' (native-res center crop, no "
                        "downscale -- removes the resolution confound on mixed-res sets).")
    p.add_argument("--no-jpeg-equalize", dest="jpeg_equalize", action="store_false", default=True,
                   help="Disable JPEG-q90 equalization of non-JPEG inputs before DIRE. "
                        "Default ON to match the FakeFlickr eval protocol (removes the "
                        "real-JPEG vs fake-PNG/WebP format confound).")
    p.add_argument("--output-ext", default="", choices=["", ".png", ".jpg"],
                   help="Force DIRE maps to be saved with this extension.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for the classifier.")
    p.add_argument("--diffusion-batch-size", type=int, default=16,
                   help="Batch size for compute_dire_single.py.")
    p.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                   help="GPU to expose to compute_dire_single.py (single device).")
    p.add_argument("--skip-dire", action="store_true",
                   help="Skip DIRE computation; assume DIRE maps already exist under "
                        "<work-dir>/dire/<gen>/{0_real,1_fake}/.")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: only run on --debug-samples images per set "
                        "to smoke-test the pipeline.")
    p.add_argument("--debug-samples", type=int, default=10,
                   help="Number of test IDs to keep in --debug mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not args.skip_dire and not args.diffusion_ckpt.is_file():
        raise FileNotFoundError(f"--diffusion-ckpt not found: {args.diffusion_ckpt}")
    if not args.test_split.is_file():
        raise FileNotFoundError(f"--test-split not found: {args.test_split}")

    test_ids = read_test_ids(args.test_split)
    print(f"Loaded {len(test_ids)} test IDs from {args.test_split}")
    if args.debug:
        test_ids = test_ids[: args.debug_samples]
        print(f"[DEBUG] truncated to {len(test_ids)} IDs")

    stage_root = args.work_dir / "stage"
    recons_root = args.work_dir / "recons"
    dire_root = args.work_dir / "dire"
    stage_root.mkdir(parents=True, exist_ok=True)
    recons_root.mkdir(parents=True, exist_ok=True)
    dire_root.mkdir(parents=True, exist_ok=True)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading classifier {args.arch} from {args.ckpt}")
    model = get_network(args.arch)
    state = torch.load(args.ckpt, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    def compute_dire_for(stage_parent: Path, dire_dir: Path, recons_dir: Path,
                         n_samples: int) -> None:
        for d in (recons_dir, dire_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        run_compute_dire(
            images_dir=stage_parent,
            recons_dir=recons_dir,
            dire_dir=dire_dir,
            diffusion_ckpt=args.diffusion_ckpt,
            num_samples=n_samples,
            batch_size=args.diffusion_batch_size,
            cuda_visible_devices=args.cuda_visible_devices,
            jpeg_equalize=args.jpeg_equalize,
            crop_mode=args.crop_mode,
            output_ext=args.output_ext,
        )

    # The real images are shared across all generators that use the same real
    # source, so we compute their (expensive) DIRE maps only once per source.
    # "real" for normal generators, "real_rescaled" for the rescaled-real gen.
    real_sources = {("real_rescaled" if g in RESCALED_REAL_GENS else "real")
                    for g in args.generators}
    shared_real_dire: dict[str, Path] = {}
    for rn in sorted(real_sources):
        dire_dir = dire_root / f"_shared_{rn}" / rn
        shared_real_dire[rn] = dire_dir
        if args.skip_dire:
            continue
        print(f"\n=== shared reals: {rn} ===")
        stage_parent = stage_root / f"_shared_{rn}"
        n = stage_files(stage_parent, args.dataset_root / rn, test_ids, rn)
        print(f"  staged {n} real images ({rn})")
        compute_dire_for(stage_parent, dire_root / f"_shared_{rn}",
                         recons_root / f"_shared_{rn}", n)

    rows: list[dict] = []
    for gen in args.generators:
        print(f"\n=== {gen} ===")
        rn = "real_rescaled" if gen in RESCALED_REAL_GENS else "real"
        fake_dire_dir = dire_root / gen / "1_fake"
        if not args.skip_dire:
            stage_parent = stage_root / gen
            n = stage_files(stage_parent, args.dataset_root / "generated" / gen / "img",
                            test_ids, "1_fake")
            print(f"  staged {n} fake images")
            compute_dire_for(stage_parent, dire_root / gen, recons_root / gen, n)

        real_dire_dir = shared_real_dire[rn]
        if not real_dire_dir.is_dir() or not fake_dire_dir.is_dir():
            raise RuntimeError(
                f"DIRE maps missing (real={real_dire_dir}, fake={fake_dire_dir})")

        metrics = evaluate_generator(
            model, real_dire_dir, fake_dire_dir,
            cfg_load_size=args.load_size,
            cfg_crop_size=args.crop_size,
            aug_norm=args.aug_norm,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"  {gen}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in metrics.items()))
        rows.append({"generator": gen, **metrics})

    fieldnames = ["generator", "ACC", "AP", "R_ACC", "F_ACC", "N_real", "N_fake"]
    with args.results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {args.results_csv}")


if __name__ == "__main__":
    main()
