# CycleGAN Monet ↔ Foto — Kelompok 4

CycleGAN from scratch (PyTorch) untuk translasi gaya lukisan Monet ↔ foto pemandangan (dataset monet2photo, 128×128).

| Dokumen | Isi |
| --- | --- |
| `LAPORAN_AKHIR.md` | Laporan akhir kelompok |
| `LOG_INTERAKSI.md` | Log interaksi LLM seluruh anggota |
| `README_ANGGOTA3.md` | Detail training loop dan eksperimen |
| `hasil/` | Grafik, tabel, dan contoh gambar hasil eksperimen |

## Setup lingkungan

Lingkungan yang sudah teruji: Windows 11, Python 3.13.2, PyTorch 2.11.0+cu128,
NVIDIA GeForce RTX 3050 Laptop GPU 4 GB.

> Perintah di bawah dipisah per OS. Jangan mencampur perintah Windows dan macOS/Linux.
> Folder `.venv/` dan `data/` tidak di-commit; setiap anggota membuatnya sendiri.

### 0. Prasyarat
- Python 3.13 (cek: `python --version`, atau `py --version` di Windows)
- **Windows/Linux dengan GPU NVIDIA:** jalankan `nvidia-smi`. Pastikan GPU terdeteksi dan
  "CUDA Version" di pojok kanan atas ≥ 12.8. Jika lebih rendah, perbarui driver NVIDIA dulu.

### 1. Buat dan aktifkan virtual environment
Jalankan dari folder root repo.

**Windows (Command Prompt / cmd)**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Jika muncul *"running scripts is disabled on this system"*, jalankan dulu perintah berikut
(hanya berlaku untuk jendela PowerShell yang sedang terbuka), lalu ulangi aktivasi:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Venv aktif jika prompt diawali `(.venv)`. Pastikan Python yang dipakai berasal dari venv:
```
python -c "import sys; print(sys.executable)"
```
Path yang tercetak harus berada di dalam folder `.venv`.

### 2. Pasang PyTorch (perintahnya BERBEDA per platform)
Wajib dalam keadaan venv aktif.

**Windows / Linux dengan GPU NVIDIA**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Di Windows, `pip install torch` tanpa `--index-url` akan memasang versi **CPU saja**
(PyPI hanya menyediakan versi CPU untuk Windows).

**macOS** (tidak ada CUDA; di Apple Silicon kode memakai GPU MPS lewat pemilihan device CUDA → MPS → CPU, cukup untuk uji cepat, tidak untuk training penuh)
```
pip install torch torchvision
```

**Sudah terlanjur memasang versi CPU?** Hapus dulu, karena pip tidak akan menggantinya:
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Pasang dependensi lain
```
pip install pandas matplotlib pillow
```

### 4. Cek PyTorch dan GPU
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
| Output | Arti |
|---|---|
| `2.11.0+cu128 True` | Benar (Windows/Linux dengan GPU) |
| `... False` dan versi diakhiri `+cpu` | Terpasang versi CPU, ulangi langkah 2 bagian "Sudah terlanjur" |
| `... False` dan versi diakhiri `+cu128` | Driver NVIDIA bermasalah, cek `nvidia-smi` |
| `ModuleNotFoundError: No module named 'torch'` | Venv belum aktif, atau torch belum dipasang di venv ini |

### 5. Unduh dataset
```
python download_data.py
```
Hasilnya adalah folder `data/monet2photo/` berisi `trainA`, `trainB`, `testA`, `testB`.
Cek dengan `dir data\monet2photo` (Windows) atau `ls data/monet2photo` (macOS/Linux).

### 6. Smoke test
```
python smoke_test.py
```
Hasil yang diharapkan: `6 OK, 0 GAGAL`. Log tersimpan di `logs/smoke_test_<tanggal_jam>.txt`.
Lampirkan file log ini saat melaporkan masalah.

### Masalah umum
| Error | Penyebab | Solusi |
|---|---|---|
| `'source' is not recognized ...` | `source` hanya ada di bash/zsh (macOS/Linux) | Pakai perintah aktivasi Windows di langkah 1 |
| `running scripts is disabled on this system` | Kebijakan eksekusi PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `python` membuka Microsoft Store / tidak dikenali | Alias Windows Store, bukan Python sungguhan | Pakai `py -3.13 -m venv .venv`, atau matikan alias di *Settings → Apps → Advanced app settings → App execution aliases* |
| `No module named 'torch'` | Venv tidak aktif / torch belum dipasang di venv | Aktifkan venv, lalu ulangi langkah 2 |
| `torch.cuda.is_available()` = `False` | Versi CPU terpasang atau driver lama | Lihat tabel di langkah 4 |
| Folder `data\monet2photo` tidak ditemukan | Dataset belum diunduh | `python download_data.py` |

## Menjalankan eksperimen

```
python run_experiments.py --epochs 10 --max_iters 400
```
Hasil (grafik, tabel, contoh gambar) tersimpan di `outputs/`; salinan yang dipakai di laporan ada di `hasil/`.
