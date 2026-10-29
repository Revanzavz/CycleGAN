"""
dataset.py
Dataset unpaired untuk CycleGAN (monet2photo) dan fungsi pembuat DataLoader.

Konvensi proyek:
- Domain A = lukisan Monet (trainA/testA), domain B = foto (trainB/testB)
- Tensor gambar berukuran (3, 128, 128) dengan rentang [-1, 1]

Struktur folder yang diharapkan (dibuat oleh download_data.py):
    <root>/trainA  <root>/trainB  <root>/testA  <root>/testB

Uji cepat: python dataset.py
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
UKURAN_RESIZE_TRAIN = 143  # diperbesar sedikit dulu, lalu di-crop acak ke 128
UKURAN_GAMBAR = 128        # ukuran akhir (3, 128, 128)
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png"}  # sama dengan download_data.py

DIR_PROYEK = Path(__file__).resolve().parent
ROOT_BAWAAN = DIR_PROYEK / "data" / "monet2photo"
OUTPUT_DIR = DIR_PROYEK / "outputs"


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def buat_transform(train):
    """Transform training (augmentasi) atau test (deterministik).
    Keduanya berakhir di tensor float32 (3, 128, 128) dengan rentang [-1, 1]."""
    # (x - 0.5) / 0.5 mengubah rentang [0, 1] dari ToTensor menjadi [-1, 1]
    normalisasi = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    bicubic = T.InterpolationMode.BICUBIC

    if train:
        return T.Compose([
            T.Resize((UKURAN_RESIZE_TRAIN, UKURAN_RESIZE_TRAIN), interpolation=bicubic),
            T.RandomCrop(UKURAN_GAMBAR),          # posisi crop acak: 16 x 16 kemungkinan
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalisasi,
        ])

    # Test: tanpa unsur acak supaya hasil bisa diulang
    return T.Compose([
        T.Resize((UKURAN_GAMBAR, UKURAN_GAMBAR), interpolation=bicubic),
        T.ToTensor(),
        normalisasi,
    ])


def _daftar_gambar(folder):
    """Daftar file gambar di dalam folder, diurutkan agar urutannya konsisten."""
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {folder}\n"
            "Jalankan 'python download_data.py' terlebih dahulu."
        )
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EKSTENSI_GAMBAR
    )
    if not paths:
        raise FileNotFoundError(f"Tidak ada gambar (.jpg/.jpeg/.png) di folder: {folder}")
    return paths


def _baca_rgb(path):
    """Baca gambar dan paksa menjadi RGB 3 kanal (grayscale, RGBA, palette ikut tertangani).
    Memakai 'with' agar file langsung ditutup setelah dibaca."""
    with Image.open(path) as img:
        return img.convert("RGB")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class UnpairedDataset(Dataset):
    """
    Dataset unpaired dua domain untuk CycleGAN.
    Setiap item berupa dict {"A": tensor Monet, "B": tensor foto}, masing-masing (3, 128, 128).

    - Panjang dataset = jumlah gambar A, sehingga satu epoch = satu kali melewati semua A.
    - random_pairing=True (train): gambar B dipilih acak untuk setiap A.
    - random_pairing=False (test): B dipilih deterministik (index % jumlah B)
      agar evaluasi dan visualisasi bisa diulang.
    """

    def __init__(self, root, split="train", transform=None, random_pairing=True):
        if split not in ("train", "test"):
            raise ValueError(f"split harus 'train' atau 'test', bukan {split!r}")

        root = Path(root)
        self.paths_a = _daftar_gambar(root / f"{split}A")  # lukisan Monet
        self.paths_b = _daftar_gambar(root / f"{split}B")  # foto
        self.transform = transform if transform is not None else buat_transform(split == "train")
        self.random_pairing = random_pairing

    def __len__(self):
        return len(self.paths_a)

    def _indeks_b(self, index):
        """Pilih indeks gambar B untuk gambar A ke-index."""
        if self.random_pairing:
            # Acak seragam dan tidak bergantung pada index. Pada data unpaired tidak ada
            # pasangan yang "benar", jadi pemasangan hanya cara membentuk batch.
            # Di dalam worker DataLoader, modul random di-seed ulang per worker.
            return random.randint(0, len(self.paths_b) - 1)
        return index % len(self.paths_b)

    def __getitem__(self, index):
        img_a = _baca_rgb(self.paths_a[index])
        img_b = _baca_rgb(self.paths_b[self._indeks_b(index)])
        # Transform dipanggil terpisah, jadi crop dan flip A dan B saling independen
        return {"A": self.transform(img_a), "B": self.transform(img_b)}


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
def get_dataloaders(root, batch_size=1, num_workers=2, pin_memory=True):
    """Mengembalikan (train_loader, test_loader).
    Di Windows, pemanggil harus berada di dalam blok if __name__ == "__main__"."""
    train_ds = UnpairedDataset(root, split="train", random_pairing=True)
    test_ds = UnpairedDataset(root, split="test", random_pairing=False)

    # Worker dipertahankan antar epoch (di Windows, membuat worker baru itu lambat)
    persisten = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # ukuran batch selalu sama
        persistent_workers=persisten,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persisten,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Pengujian cepat: python dataset.py
# ---------------------------------------------------------------------------
def _periksa_batch(nama, batch):
    for kunci in ("A", "B"):
        x = batch[kunci]
        print(f"[{nama}] {kunci}: shape={tuple(x.shape)}, dtype={x.dtype}, "
              f"min={x.min():.3f}, max={x.max():.3f}")
        assert x.dim() == 4 and x.shape[1:] == (3, UKURAN_GAMBAR, UKURAN_GAMBAR), \
            "Bentuk batch harus (N, 3, 128, 128)"
        assert x.dtype == torch.float32, "Tipe data harus float32"
        assert x.min() >= -1.0 and x.max() <= 1.0, "Nilai harus berada di [-1, 1]"


def _uji_cepat(args):
    train_loader, test_loader = get_dataloaders(args.root, args.batch_size, args.num_workers)
    train_ds, test_ds = train_loader.dataset, test_loader.dataset

    print(f"Root dataset : {Path(args.root).resolve()}")
    print(f"Train        : A={len(train_ds.paths_a)}, B={len(train_ds.paths_b)}, "
          f"iterasi per epoch={len(train_loader)}")
    print(f"Test         : A={len(test_ds.paths_a)}, B={len(test_ds.paths_b)}, "
          f"iterasi per epoch={len(test_loader)}")

    # Pemasangan: acak pada train, deterministik pada test
    indeks_train = {train_ds._indeks_b(0) for _ in range(30)}
    print(f"Train: 30 kali memilih B untuk A ke-0 -> {len(indeks_train)} indeks B berbeda")
    assert len(indeks_train) > 1, "Pemasangan train seharusnya acak"
    assert test_ds._indeks_b(5) == test_ds._indeks_b(5), "Pemasangan test seharusnya tetap"
    print("Pemasangan  : train acak, test deterministik (OK)")

    # Bentuk, tipe, dan rentang nilai pada satu batch dari tiap loader
    batch_train = next(iter(train_loader))
    _periksa_batch("train", batch_train)
    _periksa_batch("test ", next(iter(test_loader)))

    # Grid contoh: baris atas = Monet (A), baris bawah = foto (B), setelah augmentasi train
    a, b = batch_train["A"], batch_train["B"]
    semua = (torch.cat([a, b], dim=0) * 0.5 + 0.5).clamp(0, 1)  # kembali ke [0, 1]
    grid = make_grid(semua, nrow=a.size(0), padding=2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tujuan = OUTPUT_DIR / "sample_data.png"
    save_image(grid, tujuan)
    print(f"Grid contoh disimpan di: {tujuan}")
    print("Semua pengecekan lolos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji cepat dataset.py")
    parser.add_argument("--root", default=str(ROOT_BAWAAN), help="folder monet2photo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    _uji_cepat(parser.parse_args())