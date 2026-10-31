"""
utils.py
Fungsi pendukung training CycleGAN:
- ImageBuffer      : menyimpan 50 gambar palsu terdahulu untuk melatih diskriminator.
- save/load_checkpoint : menyimpan dan memuat semua model, optimizer, scheduler, scaler.
- save_sample_grid : menyimpan grid real A | fake B | rec A | real B | fake A | rec B.
- CSVLogger        : mencatat loss per epoch ke file CSV.
- linear_decay_lambda : aturan lr konstan separuh pertama, lalu turun linear ke nol.
- set_seed         : supaya eksperimen bisa diulang.
"""

import csv
import os
import random

import numpy as np
import torch
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Image buffer (history of generated images)
# ---------------------------------------------------------------------------
class ImageBuffer:
    """Menyimpan sampai `max_size` gambar palsu dari iterasi-iterasi sebelumnya.

    Saat diskriminator dilatih, setiap gambar palsu baru punya peluang 50% ditukar
    dengan gambar lama dari buffer. Dengan begitu D tidak hanya melihat keluaran
    generator terbaru, sehingga D tidak "kejar-kejaran" dengan G dan training
    lebih stabil (osilasi berkurang).
    """

    def __init__(self, max_size: int = 50):
        assert max_size > 0, "Ukuran buffer harus > 0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        # Pastikan gambar tidak membawa graf komputasi generator
        images = images.detach()
        result = []
        for img in images:  # proses per gambar dalam batch
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                # Buffer belum penuh: simpan dan pakai gambar baru
                self.data.append(img.clone())
                result.append(img)
            elif random.random() > 0.5:
                # 50%: pakai gambar lama, ganti posisinya dengan gambar baru
                idx = random.randint(0, self.max_size - 1)
                old = self.data[idx].clone()
                self.data[idx] = img.clone()
                result.append(old)
            else:
                # 50%: pakai gambar baru apa adanya
                result.append(img)
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, epoch: int, models: dict, optimizers: dict,
                    schedulers: dict = None, scalers: dict = None, extra: dict = None):
    """Simpan state semua komponen ke satu file .pth.

    `models`, `optimizers`, dst. berupa dict nama -> objek, misalnya
    {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch": epoch,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "schedulers": {k: s.state_dict() for k, s in (schedulers or {}).items()},
        "scalers": {k: s.state_dict() for k, s in (scalers or {}).items()},
        "extra": extra or {},
    }
    # Tulis ke file sementara dulu, baru diganti: file lama tidak rusak kalau proses mati di tengah
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, models: dict, optimizers: dict = None,
                    schedulers: dict = None, scalers: dict = None, map_location="cpu") -> dict:
    """Muat checkpoint ke objek yang sudah dibuat. Mengembalikan isi checkpoint
    (misalnya ckpt["epoch"] untuk melanjutkan training)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    for k, m in models.items():
        m.load_state_dict(ckpt["models"][k])
    for group_name, objs in (("optimizers", optimizers), ("schedulers", schedulers), ("scalers", scalers)):
        for k, o in (objs or {}).items():
            if k in ckpt.get(group_name, {}):
                o.load_state_dict(ckpt[group_name][k])
    return ckpt


# ---------------------------------------------------------------------------
# Grid contoh gambar
# ---------------------------------------------------------------------------
def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Ubah rentang [-1, 1] kembali ke [0, 1] untuk disimpan sebagai gambar."""
    return (x.float().clamp(-1, 1) + 1) / 2


def save_sample_grid(path: str, real_A, fake_B, rec_A, real_B, fake_A, rec_B, max_rows: int = 4):
    """Simpan grid dengan 6 kolom per baris:
    real A (Monet) | fake B (foto palsu) | rec A | real B (foto) | fake A (Monet palsu) | rec B
    Setiap baris adalah satu contoh dari batch.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tensors = [real_A, fake_B, rec_A, real_B, fake_A, rec_B]
    n = min(max_rows, *[t.size(0) for t in tensors])
    rows = []
    for i in range(n):
        for t in tensors:
            rows.append(denormalize(t[i].detach().cpu()))
    grid = torch.stack(rows, dim=0)
    vutils.save_image(grid, path, nrow=6, padding=2)


def fixed_test_samples(test_loader, n: int = 4):
    """Ambil n gambar pertama dari dataset test (tanpa acak) sebagai contoh tetap,
    tidak bergantung pada batch_size. Mengembalikan (real_A, real_B)."""
    ds = test_loader.dataset
    items = [ds[i] for i in range(min(n, len(ds)))]
    if isinstance(items[0], dict):
        return (torch.stack([it["A"] for it in items]), torch.stack([it["B"] for it in items]))
    return (torch.stack([it[0] for it in items]), torch.stack([it[1] for it in items]))


# ---------------------------------------------------------------------------
# Logger CSV
# ---------------------------------------------------------------------------
class CSVLogger:
    """Mencatat satu baris per epoch ke file CSV. Kalau training dilanjutkan
    (resume), baris baru ditambahkan ke file lama tanpa menulis ulang header."""

    def __init__(self, path: str, fieldnames: list, resume: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not (resume and os.path.exists(path)):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Jadwal learning rate
# ---------------------------------------------------------------------------
def linear_decay_lambda(total_epochs: int):
    """Faktor pengali lr untuk torch.optim.lr_scheduler.LambdaLR.

    Separuh pertama epoch: faktor 1.0 (lr konstan).
    Separuh kedua: turun linear sehingga setelah epoch terakhir lr menjadi 0.
    Contoh total 10 epoch: epoch 0-4 -> 1.0, epoch 5 -> 1.0, 6 -> 0.8, ..., 9 -> 0.2, selesai -> 0.
    """
    half = total_epochs // 2
    decay_len = max(1, total_epochs - half)

    def rule(epoch: int) -> float:
        return max(0.0, 1.0 - max(0, epoch - half) / decay_len)

    return rule


def pick_device():
    """Pilih perangkat: GPU NVIDIA (cuda) > GPU Apple Silicon (mps) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Uji ImageBuffer
    buf = ImageBuffer(max_size=50)
    for _ in range(60):
        out = buf.push_and_pop(torch.randn(2, 3, 128, 128))
    print("Isi buffer:", len(buf), "| shape output:", tuple(out.shape))
    # Uji jadwal lr
    rule = linear_decay_lambda(10)
    print("Faktor lr per epoch:", [round(rule(e), 2) for e in range(11)])
