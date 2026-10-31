"""
evaluate.py
Evaluasi CycleGAN pada data TEST dengan metrik cycle reconstruction error:
  - rec_err_A = rata-rata L1( real_A , G_BA(G_AB(real_A)) )   (Monet -> Foto -> Monet)
  - rec_err_B = rata-rata L1( real_B , G_AB(G_BA(real_B)) )   (Foto -> Monet -> Foto)
Nilai dihitung pada rentang piksel [-1, 1]. Makin kecil makin baik (gambar bisa
dikembalikan ke aslinya, artinya informasi isi gambar tidak hilang).

Catatan: metrik ini mengukur KONSISTENSI, bukan seberapa "Monet" hasilnya. Error yang
sangat kecil bisa juga berarti generator hampir tidak mengubah gambar, jadi selalu
bandingkan dengan grid contoh gambar.

Contoh:
  python evaluate.py --data_root data/monet2photo --experiments cyc10_id5_bs1 cyc5_id5_bs1
  python evaluate.py --data_root data/monet2photo --all
"""

import argparse
import csv
import glob
import os

import torch

from dataset import get_dataloaders
from models import Generator
from utils import fixed_test_samples, pick_device, save_sample_grid


@torch.no_grad()
def cycle_reconstruction_error(G_AB, G_BA, loader, device, max_batches: int = 0):
    """Hitung rata-rata L1 rekonstruksi pada seluruh loader test.

    Karena testA (~120 gambar) dan testB (~750 gambar) jumlahnya berbeda, error
    dirata-rata per gambar, bukan per batch.
    """
    G_AB.eval()
    G_BA.eval()
    total_A = total_B = 0.0
    count_A = count_B = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        real_A, real_B = (batch["A"], batch["B"]) if isinstance(batch, dict) else (batch[0], batch[1])
        real_A, real_B = real_A.to(device), real_B.to(device)
        rec_A = G_BA(G_AB(real_A)).float()
        rec_B = G_AB(G_BA(real_B)).float()
        # L1 per gambar: rata-rata atas (C, H, W), lalu dijumlah atas batch
        total_A += (rec_A - real_A).abs().mean(dim=(1, 2, 3)).sum().item()
        total_B += (rec_B - real_B).abs().mean(dim=(1, 2, 3)).sum().item()
        count_A += real_A.size(0)
        count_B += real_B.size(0)
    err_A = total_A / max(1, count_A)
    err_B = total_B / max(1, count_B)
    return {"rec_err_A": err_A, "rec_err_B": err_B, "rec_err_mean": (err_A + err_B) / 2}


def find_checkpoint(exp_dir: str):
    """Pilih checkpoint terakhir: checkpoint_latest.pth, atau checkpoint_epoch_XXX terbesar."""
    latest = os.path.join(exp_dir, "checkpoint_latest.pth")
    if os.path.exists(latest):
        return latest
    numbered = sorted(glob.glob(os.path.join(exp_dir, "checkpoint_epoch_*.pth")))
    return numbered[-1] if numbered else None


def load_generators(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G_AB, G_BA = Generator().to(device), Generator().to(device)
    G_AB.load_state_dict(ckpt["models"]["G_AB"])
    G_BA.load_state_dict(ckpt["models"]["G_BA"])
    return G_AB, G_BA, ckpt.get("epoch", -1) + 1


def evaluate_experiment(exp_name, args, test_loader, device):
    exp_dir = os.path.join(args.output_dir, exp_name)
    ckpt_path = find_checkpoint(exp_dir)
    if ckpt_path is None:
        print(f"[LEWATI] {exp_name}: checkpoint tidak ditemukan di {exp_dir}")
        return None
    G_AB, G_BA, epoch = load_generators(ckpt_path, device)
    res = cycle_reconstruction_error(G_AB, G_BA, test_loader, device, args.max_batches)
    res.update({"experiment": exp_name, "epoch": epoch})
    print(f"[EVAL] {exp_name} (epoch {epoch}): rec_err_A={res['rec_err_A']:.4f} "
          f"rec_err_B={res['rec_err_B']:.4f} rata-rata={res['rec_err_mean']:.4f}")

    # Simpan satu grid hasil test untuk laporan
    real_A, real_B = fixed_test_samples(test_loader, 4)
    real_A, real_B = real_A.to(device), real_B.to(device)
    with torch.no_grad():
        fake_B, fake_A = G_AB(real_A), G_BA(real_B)
        rec_A, rec_B = G_BA(fake_B), G_AB(fake_A)
    save_sample_grid(os.path.join(exp_dir, "test_samples.png"), real_A, fake_B, rec_A, real_B, fake_A, rec_B)
    return res


def main():
    p = argparse.ArgumentParser(description="Evaluasi cycle reconstruction error pada data test")
    p.add_argument("--data_root", type=str, default="data/monet2photo")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--experiments", nargs="*", default=[], help="nama eksperimen yang dievaluasi")
    p.add_argument("--all", action="store_true", help="evaluasi semua folder di output_dir")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=0, help="0 = seluruh data test")
    p.add_argument("--result_csv", type=str, default=None,
                   help="default: <output_dir>/eval_results.csv")
    args = p.parse_args()

    device = pick_device()
    names = list(args.experiments)
    if args.all:
        names = sorted(d for d in os.listdir(args.output_dir)
                       if os.path.isdir(os.path.join(args.output_dir, d)))
    if not names:
        p.error("Sebutkan --experiments <nama ...> atau pakai --all")

    _, test_loader = get_dataloaders(args.data_root, args.batch_size, args.num_workers)
    result_csv = args.result_csv or os.path.join(args.output_dir, "eval_results.csv")

    # Gabungkan dengan hasil lama supaya evaluasi bisa dijalankan bertahap
    rows = {}
    if os.path.exists(result_csv):
        with open(result_csv) as f:
            rows = {r["experiment"]: r for r in csv.DictReader(f)}
    for name in names:
        res = evaluate_experiment(name, args, test_loader, device)
        if res:
            rows[name] = res

    fields = ["experiment", "epoch", "rec_err_A", "rec_err_B", "rec_err_mean"]
    with open(result_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print(f"[INFO] Hasil evaluasi disimpan di {result_csv}")


if __name__ == "__main__":
    main()
