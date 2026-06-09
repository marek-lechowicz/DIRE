"""Single-process (MPI-free) DIRE computation.

This is a drop-in replacement for ``compute_dire.py`` for machines with a
single GPU and no MPI runtime.  ``compute_dire.py`` depends on ``mpi4py`` and
``mpiexec`` (via ``guided_diffusion.dist_util`` / ``image_datasets`` /
``logger``); here we avoid all three and run a plain single-GPU loop.

The DIRE map produced is identical in definition to the original:

    latent  = ddim_reverse_sample_loop(model, image)      # image -> noise
    recon   = ddim_sample_loop(model, latent)             # noise -> recon
    dire    = |image - recon|        (images in [-1, 1])
    dire_uint8 = clamp(dire * 255 / 2, 0, 255)

Inputs/outputs mirror ``compute_dire.py``:

  --images_dir / --recons_dir / --dire_dir   (with --has_subfolder True the
  per-subfolder layout, e.g. 0_real/ and 1_fake/, is preserved on output).

Model/diffusion flags match guided-diffusion's 256x256 unconditional ADM
checkpoint; sampling defaults to DDIM-20 (``--timestep_respacing ddim20
--use_ddim True``), the configuration used by the DIRE paper/repo.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def list_images(data_dir: str) -> list[str]:
    out: list[str] = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if os.path.isdir(full):
            out.extend(list_images(full))
        elif os.path.splitext(entry)[1].lower() in IMG_EXTS:
            out.append(full)
    return out


def center_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    """Resize short side to image_size, then center-crop (guided-diffusion default).

    This downscales high-res images, which conflates downscale-ratio with
    generative origin (a confound on mixed-resolution sets like FakeFlickr).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def native_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    """Center-crop a native-resolution image_size patch -- NO downscaling.

    Avoids the resolution/downscale confound: every image contributes a
    native-sampling-rate crop. Only images whose short side is < image_size
    are minimally up-scaled (short side -> image_size) so a crop is possible.
    """
    w, h = pil_image.size
    if min(w, h) < image_size:
        scale = image_size / min(w, h)
        pil_image = pil_image.resize((round(w * scale), round(h * scale)), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


CROP_FNS = {"resize": center_crop_arr, "crop": native_crop_arr}


class ReverseDataset(Dataset):
    def __init__(self, image_paths: list[str], resolution: int, jpeg_equalize: bool = True,
                 crop_mode: str = "resize"):
        self.paths = image_paths
        self.resolution = resolution
        self.crop_fn = CROP_FNS[crop_mode]
        # When True, re-encode non-JPEG inputs (PNG/WebP) to JPEG quality 90,
        # mirroring the FakeFlickr eval protocol (resnet50_wandb_pipeline/data.py).
        # This equalises the format/compression bias so DIRE is computed from
        # uniformly JPEG-compressed photos -- original JPEG reals pass through
        # unchanged, lossless fakes/rescaled-reals get the same q=90 compression.
        self.jpeg_equalize = jpeg_equalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        if self.jpeg_equalize and not path.lower().endswith((".jpg", ".jpeg")):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:
            with open(path, "rb") as f:
                img = Image.open(f)
                img.load()
            img = img.convert("RGB")
        arr = self.crop_fn(img, self.resolution).astype(np.float32) / 127.5 - 1
        arr = th.from_numpy(np.transpose(arr, [2, 0, 1]))
        return arr, path


def reshape_image(imgs: th.Tensor, image_size: int) -> th.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        imgs = transforms.CenterCrop(image_size)(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main() -> None:
    args = create_argparser().parse_args()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)

    print("[compute_dire_single] creating model and diffusion ...", flush=True)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    paths = list_images(args.images_dir)
    if args.num_samples is not None and args.num_samples > 0:
        paths = paths[: args.num_samples]
    print(f"[compute_dire_single] {len(paths)} images from {args.images_dir}", flush=True)
    print(f"[compute_dire_single] jpeg_equalize={args.jpeg_equalize} crop_mode={args.crop_mode} "
          f"(non-JPEG->JPEG q90; '{args.crop_mode}' = "
          f"{'resize short-side then crop' if args.crop_mode == 'resize' else 'native-res center crop, no downscale'})",
          flush=True)
    loader = DataLoader(
        ReverseDataset(paths, args.image_size, jpeg_equalize=args.jpeg_equalize,
                       crop_mode=args.crop_mode),
        batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
    )

    reverse_fn = diffusion.ddim_reverse_sample_loop
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    done = 0
    for imgs, batch_paths in tqdm(loader, dynamic_ncols=True):
        imgs = imgs.to(device)
        imgs = reshape_image(imgs, args.image_size)
        bs = imgs.shape[0]
        model_kwargs = {}
        if args.class_cond:
            model_kwargs["y"] = th.randint(0, NUM_CLASSES, (bs,), device=device)

        latent = reverse_fn(
            model, (bs, 3, args.image_size, args.image_size), noise=imgs,
            clip_denoised=args.clip_denoised, model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        recons = sample_fn(
            model, (bs, 3, args.image_size, args.image_size), noise=latent,
            clip_denoised=args.clip_denoised, model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        dire = th.abs(imgs - recons)
        recons_u8 = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        dire_u8 = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()

        for i in range(bs):
            if args.has_subfolder:
                sub = batch_paths[i].split("/")[-2]
                recons_save_dir = os.path.join(args.recons_dir, sub)
                dire_save_dir = os.path.join(args.dire_dir, sub)
            else:
                recons_save_dir = args.recons_dir
                dire_save_dir = args.dire_dir
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)
            # BUG / DATA LEAKAGE IDENTIFIED ---
            # Using the original image's extension (fn = os.path.basename(batch_paths[i]))
            # causes cv2.imwrite to apply the same compression algorithm to the DIRE map 
            # (e.g. .jpg = lossy JPEG, .png = lossless PNG, .webp = lossy WebP).
            # This allows the downstream ResNet classifier to trivially differentiate 
            # Real (often .jpg) vs Fake (often .webp/.png) strictly by detecting the 
            # compression artifacts introduced HERE, completely ignoring DIRE features.
            # 
            # Fix: Force the extension to be uniformly lossless (e.g., .png) using --output_ext
            if args.output_ext:
                fn = os.path.splitext(os.path.basename(batch_paths[i]))[0] + args.output_ext
            else:
                fn = os.path.basename(batch_paths[i])
            cv2.imwrite(os.path.join(dire_save_dir, fn), cv2.cvtColor(dire_u8[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(recons_save_dir, fn), cv2.cvtColor(recons_u8[i], cv2.COLOR_RGB2BGR))
        done += bs
    print(f"[compute_dire_single] finished {done} images", flush=True)


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        images_dir="",
        recons_dir="",
        dire_dir="",
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
        jpeg_equalize=True,
        crop_mode="resize",  # "resize" (guided-diffusion default) or "crop" (native-res, no downscale)
        output_ext="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
