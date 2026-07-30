import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models import LowLightdenoiser, RestorationModule, UNet


def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]


def pad_image(x: torch.Tensor, multiple: int = 16):
    """Aggiunge padding se le dimensioni della foto del telefono non sono divisibili per 16."""
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x, h, w


@torch.inference_mode()
def main(checkpoint_path: str, input_path: str, output_dir: str, base_features: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Ricostruisci il modello con le base_features corrette
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_features=base_features,
        upsample_mode="pixel_shuffle",
        decoder_dropout=0.0,
    )
    denoiser = LowLightdenoiser(base_model=unet)

    # 2. Carica il checkpoint Lightning
    module = RestorationModule.load_from_checkpoint(
        checkpoint_path,
        model=denoiser,
        map_location=device,
        strict=False,
    )
    module.eval()
    module.to(device)

    # 3. Gestione input (singola immagine o cartella intera)
    inp = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    if inp.is_file():
        image_files = [inp]
    elif inp.is_dir():
        image_files = [f for f in inp.iterdir() if f.suffix.lower() in extensions]
    else:
        raise ValueError(f"Input non valido: {input_path}")

    if not image_files:
        print("Nessuna immagine trovata nell'input fornito!")
        return

    # 4. Inferenza sulle foto
    print(f"Processando {len(image_files)} immagine/i su {device}...")

    for img_path in tqdm(image_files, desc="Enhancing images"):
        x = load_image(img_path).to(device)

        # Gestisci risoluzione arbitraria dello smartphone
        x_padded, orig_h, orig_w = pad_image(x, multiple=16)

        # Forward pass
        pred = module(x_padded)

        # Rimuovi il padding per tornare alla risoluzione originale
        pred = pred[:, :, :orig_h, :orig_w]
        pred = torch.clamp(pred, 0.0, 1.0)

        # Salva l'output
        out_filename = f"enhanced_{img_path.stem}.jpg"
        pred_pil = transforms.ToPILImage()(pred.squeeze(0).cpu())
        pred_pil.save(out_dir / out_filename, quality=95)

    print(f"\nImmagini salvate in: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference su foto personalizzate o scattate da smartphone"
    )
    parser.add_argument("--checkpoint", required=True, help="Path al file .ckpt")
    parser.add_argument(
        "--input",
        required=True,
        help="Path a un file singola foto OPPURE a una cartella di foto",
    )
    parser.add_argument(
        "--output_dir", default="artifacts/custom_results", help="Cartella di output"
    )
    parser.add_argument(
        "--base_features",
        type=int,
        default=64,
        help="Numero di base features U-Net (es. 48 o 64)",
    )
    args = parser.parse_args()

    main(args.checkpoint, args.input, args.output_dir, args.base_features)