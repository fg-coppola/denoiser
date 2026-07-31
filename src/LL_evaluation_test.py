import argparse
import re
from pathlib import Path

import torch
import torchmetrics.functional.image as FM
from PIL import Image
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

from models import LowLightdenoiser, RestorationModule, UNet


def find_matching_high_file(low_file: Path, high_dir: Path) -> Path | None:
    """Searches for the corresponding file in high_dir by trying various naming conventions."""
    # 1. Identical name
    direct = high_dir / low_file.name
    if direct.exists():
        return direct

    # 2. Common prefix replacement (e.g., low00690 -> normal00690)
    name = low_file.name
    replacements = [
        name.replace("low", "normal").replace("Low", "Normal"),
        name.replace("low", "high").replace("Low", "High"),
        name.replace("low_", "normal_").replace("Low_", "Normal_"),
        name.replace("low_", "high_").replace("Low_", "High_"),
    ]
    for rep in replacements:
        cand = high_dir / rep
        if cand.exists():
            return cand

    # 3. Match based on numeric ID in the file name (e.g., '00690')
    numbers = re.findall(r"\d+", low_file.stem)
    if numbers:
        target_num = numbers[-1]
        for h_file in high_dir.iterdir():
            if h_file.is_file():
                h_numbers = re.findall(r"\d+", h_file.stem)
                if h_numbers and h_numbers[-1] == target_num:
                    return h_file

    return None


def load_image(path: Path) -> torch.Tensor:
    """Loads an image as a PyTorch Tensor [1, 3, H, W] in range [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    return transform(img).unsqueeze(0)


def resolve_first_existing(data_dir: Path, candidates: list[tuple[str, str]]) -> tuple[Path, Path]:
    """Finds the first existing folder pair (low, high/normal) on disk."""
    for low_sub, high_sub in candidates:
        low_p = data_dir / low_sub
        high_p = data_dir / high_sub
        if low_p.exists() and high_p.exists():
            return low_p, high_p
    # If no variant exists, return the first option to trigger detailed log output
    return data_dir / candidates[0][0], data_dir / candidates[0][1]


def get_eval_datasets(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    """
    Defines dataset paths supporting common folder naming variants.
    """
    v1_candidates = [
        ("LOL_v1/eval15/low", "LOL_v1/eval15/high"),
        ("LOLv1/eval15/low", "LOLv1/eval15/high"),
        ("LOL_v1/eval15/Low", "LOL_v1/eval15/High"),
        ("LOLv1/eval15/Low", "LOLv1/eval15/High"),
    ]

    v2_real_candidates = [
        ("LOL-v2/Real_captured/Test/Low", "LOL-v2/Real_captured/Test/Normal"),
        ("LOL_v2/Real_captured/Test/Low", "LOL_v2/Real_captured/Test/Normal"),
        ("LOLv2/Real_captured/Test/Low", "LOLv2/Real_captured/Test/Normal"),
        ("LOL-v2/Real_captured/Test/low", "LOL-v2/Real_captured/Test/normal"),
        ("LOL_v2/Real_captured/Test/low", "LOL_v2/Real_captured/Test/normal"),
        ("LOLv2/Real_captured/Test/low", "LOLv2/Real_captured/Test/normal"),
        ("LOL-v2/Real/Test/Low", "LOL-v2/Real/Test/Normal"),
        ("LOL_v2/Real/Test/Low", "LOL_v2/Real/Test/Normal"),
        ("LOLv2/Real/Test/Low", "LOLv2/Real/Test/Normal"),
    ]

    v2_syn_candidates = [
        ("LOL-v2/Synthetic/Test/Low", "LOL-v2/Synthetic/Test/Normal"),
        ("LOL_v2/Synthetic/Test/Low", "LOL_v2/Synthetic/Test/Normal"),
        ("LOLv2/Synthetic/Test/Low", "LOLv2/Synthetic/Test/Normal"),
        ("LOL-v2/Synthetic/Test/low", "LOL-v2/Synthetic/Test/normal"),
        ("LOL_v2/Synthetic/Test/low", "LOL_v2/Synthetic/Test/normal"),
        ("LOLv2/Synthetic/Test/low", "LOLv2/Synthetic/Test/normal"),
    ]

    return {
        "LOL_v1": resolve_first_existing(data_dir, v1_candidates),
        "LOL_v2_Real": resolve_first_existing(data_dir, v2_real_candidates),
        "LOL_v2_Synthetic": resolve_first_existing(data_dir, v2_syn_candidates),
    }


@torch.inference_mode()
def evaluate_single_dataset(
    module: torch.nn.Module,
    low_dir: Path,
    high_dir: Path,
    out_dir: Path,
    dataset_name: str,
    device: torch.device
) -> dict | None:
    """Evaluates the model on a specific pair of low/high directories."""
    if not low_dir.exists() or not high_dir.exists():
        print(f"\n[SKIP] Directories not found for {dataset_name}:")
        print(f"  - Low:  {low_dir} (Exists: {low_dir.exists()})")
        print(f"  - High: {high_dir} (Exists: {high_dir.exists()})")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    low_files = []
    for ext in valid_exts:
        low_files.extend(sorted(low_dir.glob(ext)))

    # Remove duplicates while preserving order
    low_files = list(dict.fromkeys(low_files))

    if not low_files:
        print(f"\n[SKIP] No images found in: {low_dir}")
        return None

    all_psnr_pred, all_ssim_pred = [], []
    all_psnr_base, all_ssim_base = [], []

    pbar = tqdm(low_files, desc=f"Testing {dataset_name}", leave=False)
    for low_file in pbar:
        high_file = find_matching_high_file(low_file, high_dir)
        if high_file is None:
            continue

        x = load_image(low_file).to(device)
        y = load_image(high_file).to(device)

        # Inference
        pred = module(x)
        pred = torch.clamp(pred, 0.0, 1.0)

        # Visual output saving
        save_image(pred.squeeze(0), out_dir / low_file.name)

        # Metric computation
        pred_psnr = FM.peak_signal_noise_ratio(pred, y, data_range=1.0)
        pred_ssim = FM.structural_similarity_index_measure(pred, y, data_range=1.0)
        base_psnr = FM.peak_signal_noise_ratio(x, y, data_range=1.0)
        base_ssim = FM.structural_similarity_index_measure(x, y, data_range=1.0)

        all_psnr_pred.append(pred_psnr.item())
        all_ssim_pred.append(pred_ssim.item())
        all_psnr_base.append(base_psnr.item())
        all_ssim_base.append(base_ssim.item())

    if not all_psnr_pred:
        print(f"\n[SKIP] No match found between file names in Low and High for {dataset_name}.")
        return None

    avg_psnr_base = sum(all_psnr_base) / len(all_psnr_base)
    avg_psnr_pred = sum(all_psnr_pred) / len(all_psnr_pred)
    avg_ssim_base = sum(all_ssim_base) / len(all_ssim_base)
    avg_ssim_pred = sum(all_ssim_pred) / len(all_ssim_pred)

    return {
        "count": len(all_psnr_pred),
        "psnr_base": avg_psnr_base,
        "psnr_pred": avg_psnr_pred,
        "psnr_delta": avg_psnr_pred - avg_psnr_base,
        "ssim_base": avg_ssim_base,
        "ssim_pred": avg_ssim_pred,
        "ssim_delta": avg_ssim_pred - avg_ssim_base,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Low-Light Enhancement Model")
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt file")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--output_dir", default="artifacts/eval_results", help="Output directory")
    parser.add_argument("--base_features", type=int, default=64, help="Number of base U-Net features (e.g. 48 or 64)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "lol_v1", "lol_v2_real", "lol_v2_real_captured", "lol_v2_synthetic"],
        help="Select a specific dataset or 'all' to test them all",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Architecture Reconstruction
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_features=args.base_features,
        upsample_mode="pixel_shuffle",
        decoder_dropout=0.0,
    )
    denoiser = LowLightdenoiser(base_model=unet)

    # 2. Checkpoint Loading
    print(f"Loading checkpoint from: {args.checkpoint}")
    module = RestorationModule.load_from_checkpoint(
        args.checkpoint,
        model=denoiser,
        map_location=device,
        strict=False,
    )
    module.eval()
    module.to(device)

    # 3. Dataset Preparation
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    eval_datasets = get_eval_datasets(data_path)

    if args.dataset != "all":
        key_map = {
            "lol_v1": "LOL_v1",
            "lol_v2_real": "LOL_v2_Real",
            "lol_v2_real_captured": "LOL_v2_Real",
            "lol_v2_synthetic": "LOL_v2_Synthetic",
        }
        target_key = key_map[args.dataset]
        eval_datasets = {target_key: eval_datasets[target_key]}

    results = {}

    # 4. Test Execution
    for ds_name, (low_dir, high_dir) in eval_datasets.items():
        res = evaluate_single_dataset(
            module, low_dir, high_dir, output_path / ds_name, ds_name, device
        )
        if res:
            results[ds_name] = res

    # 5. Print Results Table
    if not results:
        print("\nNo datasets were processed successfully.")
        return

    print("\n" + "=" * 80)
    print(f"{'DATASET':<18} | {'IMGS':<5} | {'PSNR IN':<8} | {'PSNR OUT':<8} | {'Δ PSNR':<8} | {'SSIM IN':<8} | {'SSIM OUT':<8} | {'Δ SSIM':<8}")
    print("=" * 80)

    total_imgs = 0
    tot_psnr_in, tot_psnr_out = 0.0, 0.0
    tot_ssim_in, tot_ssim_out = 0.0, 0.0

    for ds_name, res in results.items():
        count = res["count"]
        total_imgs += count
        tot_psnr_in += res["psnr_base"] * count
        tot_psnr_out += res["psnr_pred"] * count
        tot_ssim_in += res["ssim_base"] * count
        tot_ssim_out += res["ssim_pred"] * count

        print(
            f"{ds_name:<18} | {count:<5} | {res['psnr_base']:<8.2f} | {res['psnr_pred']:<8.2f} | "
            f"+{res['psnr_delta']:<7.2f} | {res['ssim_base']:<8.4f} | {res['ssim_pred']:<8.4f} | +{res['ssim_delta']:<7.4f}"
        )

    if len(results) > 1:
        print("-" * 80)
        mean_psnr_in = tot_psnr_in / total_imgs
        mean_psnr_out = tot_psnr_out / total_imgs
        mean_ssim_in = tot_ssim_in / total_imgs
        mean_ssim_out = tot_ssim_out / total_imgs

        print(
            f"{'GLOBAL AVERAGE':<18} | {total_imgs:<5} | {mean_psnr_in:<8.2f} | {mean_psnr_out:<8.2f} | "
            f"+{(mean_psnr_out - mean_psnr_in):<7.2f} | {mean_ssim_in:<8.4f} | {mean_ssim_out:<8.4f} | +{(mean_ssim_out - mean_ssim_in):<7.4f}"
        )
    print("=" * 80)
    print(f"Processed images saved to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()