"""
smoke_test.py - Uji integrasi cepat seluruh modul CycleGAN (Monet <-> Foto).

Konvensi yang diuji:
    G_AB : Monet (A) -> Foto (B)      D_A : diskriminator domain Monet
    G_BA : Foto (B)  -> Monet (A)     D_B : diskriminator domain Foto
    gambar (B, 3, 128, 128) di rentang [-1, 1], output D (B, 1, 14, 14) tanpa sigmoid

Pemakaian:
    python smoke_test.py                       # semua langkah
    python smoke_test.py --max_iters 10        # training sedikit lebih lama
    python smoke_test.py --skip_train          # lewati langkah training
    python smoke_test.py --no_amp              # teruskan --no_amp ke train.py

Semua output disimpan ke logs/smoke_test_<YYYY-MM-DD_HHMMSS>.txt
Exit code 0 jika semua langkah OK, 1 jika ada yang GAGAL.
"""

import argparse
import datetime as dt
import gc
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IMG_SHAPE = (3, 128, 128)
D_OUT_SHAPE = (1, 14, 14)
TOL = 1e-3  # toleransi pembulatan untuk cek rentang [-1, 1]
RE_NAN = re.compile(r"\b(nan|inf)\b", re.IGNORECASE)

# Modul tim harus bisa di-import dari folder skrip ini
sys.path.insert(0, str(ROOT_DIR))


# ----------------------------------------------------------------------------
# Infrastruktur: logger, pengecekan, runner
# ----------------------------------------------------------------------------
class Logger:
    """Mencetak ke layar sekaligus menulis ke file log (UTF-8, flush tiap baris)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:  # konsol Windows lama (cp1252) tidak kuat karakter tertentu
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        self._f.write(msg + "\n")
        self._f.flush()  # pastikan tersimpan walau skrip dihentikan paksa

    def close(self) -> None:
        self._f.close()


class LangkahDilewati(Exception):
    """Dilempar langkah yang sengaja tidak dijalankan (mis. --skip_train)."""


def cek(kondisi, pesan: str) -> None:
    """Pengganti assert (assert mati jika Python dijalankan dengan -O)."""
    if not kondisi:
        raise AssertionError(pesan)


def peringatan(ctx: dict, L: Logger, pesan: str) -> None:
    ctx["peringatan"].append(pesan)
    L.log(f"  [PERINGATAN] {pesan}")


def jalankan_langkah(no: int, nama: str, fungsi, ctx: dict, L: Logger) -> dict:
    """Bungkus satu langkah: try-except, cetak OK/GAGAL + traceback, ukur durasi."""
    L.log("")
    L.log("=" * 72)
    L.log(f"[{no}] {nama}")
    L.log("=" * 72)
    t0 = time.perf_counter()
    catatan = ""
    try:
        catatan = fungsi(ctx, L) or ""
        status = "OK"
    except LangkahDilewati as e:
        status, catatan = "DILEWATI", str(e)
    except Exception as e:  # KeyboardInterrupt sengaja tidak ditangkap
        status = "GAGAL"
        catatan = f"{type(e).__name__}: {e}".splitlines()[0][:120]
        L.log("")
        L.log("Traceback lengkap:")
        L.log(traceback.format_exc().rstrip())
    durasi = time.perf_counter() - t0
    L.log(f"--> [{no}] {status} ({durasi:.1f} s){'  ' + catatan if catatan else ''}")
    return {"no": no, "nama": nama, "status": status, "durasi": durasi, "catatan": catatan}


# ----------------------------------------------------------------------------
# Helper PyTorch
# ----------------------------------------------------------------------------
def _device(ctx: dict):
    import torch
    if "device" not in ctx:
        # Urutan sama dengan pick_device di train.py: CUDA -> MPS (Apple Silicon) -> CPU
        if torch.cuda.is_available():
            ctx["device"] = torch.device("cuda")
        elif torch.backends.mps.is_available():
            ctx["device"] = torch.device("mps")
        else:
            ctx["device"] = torch.device("cpu")
    return ctx["device"]


def _reset_vram(device) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _log_vram(device, L: Logger, label: str) -> None:
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
        puncak = torch.cuda.max_memory_allocated() / 1024**2
        L.log(f"  VRAM puncak ({label}): {puncak:.0f} MiB")
    elif device.type == "mps":
        # MPS tidak menyediakan statistik puncak, jadi yang dicatat adalah memori saat ini
        torch.mps.synchronize()
        sekarang = torch.mps.current_allocated_memory() / 1024**2
        L.log(f"  Memori MPS terpakai ({label}, bukan puncak): {sekarang:.0f} MiB")


def _bersihkan_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()  # bebaskan memori MPS sebelum train.py dijalankan
    except Exception:
        pass


def _ambil_batch(ctx: dict, L: Logger, device):
    """Pakai batch asli dari Langkah 2; jika tidak ada, buat tensor dummy di [-1, 1]."""
    import torch
    if "batch" in ctx:
        return ctx["batch"]["A"].to(device), ctx["batch"]["B"].to(device), "dataset"
    peringatan(ctx, L, "Batch dataset tidak tersedia, memakai tensor dummy acak di [-1, 1]")
    bs = ctx["batch_size"]
    gen = torch.Generator().manual_seed(0)
    a = (torch.rand(bs, *IMG_SHAPE, generator=gen) * 2 - 1).to(device)
    b = (torch.rand(bs, *IMG_SHAPE, generator=gen) * 2 - 1).to(device)
    return a, b, "dummy"


def _ringkas(x) -> str:
    return (f"shape={tuple(x.shape)} dtype={x.dtype} min={float(x.min()):+.3f} "
            f"max={float(x.max()):+.3f} mean={float(x.mean()):+.3f}")


def _cek_gambar(x, label: str, ctx: dict, L: Logger) -> None:
    """Cek satu tensor gambar terhadap konvensi proyek."""
    import torch
    cek(torch.is_tensor(x), f"{label} bukan tensor (tipe: {type(x).__name__})")
    L.log(f"  {label:<14} {_ringkas(x)}")
    cek(x.dim() == 4, f"{label}: harus 4 dimensi (B, C, H, W), didapat {x.dim()} dimensi")
    cek(x.shape[1] == 3, f"{label}: harus 3 kanal (RGB), didapat {x.shape[1]} kanal")
    cek(tuple(x.shape[1:]) == IMG_SHAPE,
        f"{label}: shape per gambar harus {IMG_SHAPE}, didapat {tuple(x.shape[1:])}")
    cek(x.dtype == torch.float32, f"{label}: dtype harus float32, didapat {x.dtype}")
    cek(bool(torch.isfinite(x).all()), f"{label}: mengandung NaN/Inf")
    cek(float(x.min()) >= -1 - TOL and float(x.max()) <= 1 + TOL,
        f"{label}: nilai di luar [-1, 1] (min={float(x.min()):.3f}, max={float(x.max()):.3f})")
    # Tanda umum lupa Normalize(0.5, 0.5): semua nilai masih di [0, 1]
    if float(x.min()) >= 0:
        peringatan(ctx, L, f"{label}: tidak ada nilai negatif, kemungkinan masih di rentang [0, 1]")


def _cek_skalar(v, label: str, L: Logger, perlu_grad: bool = True) -> float:
    import torch
    cek(torch.is_tensor(v), f"{label}: harus tensor, didapat {type(v).__name__}")
    cek(v.numel() == 1, f"{label}: harus skalar, didapat shape {tuple(v.shape)}")
    cek(bool(torch.isfinite(v).all()), f"{label}: bernilai NaN/Inf ({v.item()})")
    if perlu_grad:
        cek(v.requires_grad, f"{label}: tidak terhubung ke graf komputasi (requires_grad=False)")
    L.log(f"  {label:<26} = {v.item():.4f}")
    return v.item()


def _cek_gradien(model, label: str, L: Logger) -> None:
    """Pastikan gradien mengalir ke model: ada, terhingga, dan tidak nol semua."""
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    tanpa_grad = sum(1 for p in params if p.grad is None)
    cek(tanpa_grad < len(params), f"{label}: tidak ada parameter yang menerima gradien")
    grads = [p.grad for p in params if p.grad is not None]
    cek(all(bool(torch.isfinite(g).all()) for g in grads), f"{label}: gradien mengandung NaN/Inf")
    norma = float(torch.norm(torch.stack([g.norm() for g in grads])))
    cek(norma > 0, f"{label}: norma gradien = 0 (gradien tidak mengalir)")
    L.log(f"  grad {label:<5} norma={norma:.4e} | parameter tanpa grad: {tanpa_grad}/{len(params)}")


# ----------------------------------------------------------------------------
# Langkah-langkah uji
# ----------------------------------------------------------------------------
def langkah_lingkungan(ctx: dict, L: Logger):
    """Langkah 0: info lingkungan (siap ditempel ke prompt LLM)."""
    L.log(f"  Waktu       : {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    L.log(f"  OS          : {platform.platform()}")
    L.log(f"  Python      : {sys.version.split()[0]} ({sys.executable})")
    L.log(f"  Argumen     : batch_size={ctx['batch_size']} num_workers={ctx['num_workers']} "
          f"max_iters={ctx['max_iters']} no_amp={ctx['no_amp']}")
    L.log(f"  data_root   : {ctx['data_root']}")

    # Commit git (tidak wajib, jadi kegagalan di sini tidak menggagalkan langkah)
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                           capture_output=True, text=True, timeout=10)
        commit = r.stdout.strip() or "(bukan repo git)"
        r2 = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT_DIR,
                            capture_output=True, text=True, timeout=10)
        if r2.stdout.strip():
            commit += " (+ perubahan belum di-commit)"
    except Exception:
        commit = "(git tidak tersedia)"
    L.log(f"  Commit      : {commit}")

    import torch
    torch.manual_seed(0)  # supaya hasil smoke test bisa direproduksi
    L.log(f"  PyTorch     : {torch.__version__} | CUDA build: {torch.version.cuda} "
          f"| cuDNN: {torch.backends.cudnn.version()}")
    device = _device(ctx)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        L.log(f"  GPU         : {p.name} | VRAM total {p.total_memory / 1024**3:.2f} GB")
    elif device.type == "mps":
        L.log(f"  GPU         : Apple Silicon (MPS) | {platform.machine()}")
    else:
        peringatan(ctx, L, "CUDA dan MPS tidak tersedia, semua langkah berjalan di CPU (lambat)")
    L.log(f"  Device      : {device}")


def langkah_folder(ctx: dict, L: Logger):
    """Langkah 1: struktur folder dataset dan jumlah gambar."""
    root: Path = ctx["data_root"]
    cek(root.is_dir(), f"Folder dataset tidak ditemukan: {root} (jalankan download_data.py dulu)")
    masalah = []
    for sub in ("trainA", "trainB", "testA", "testB"):
        d = root / sub
        if not d.is_dir():
            L.log(f"  {sub:<7}: TIDAK ADA")
            masalah.append(f"{sub} tidak ada")
            continue
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        domain = "Monet" if sub.endswith("A") else "Foto"
        L.log(f"  {sub:<7}: {n:5d} gambar ({domain})")
        if n == 0:
            masalah.append(f"{sub} kosong")
    cek(not masalah, "Masalah folder: " + ", ".join(masalah))


def langkah_dataset(ctx: dict, L: Logger):
    """Langkah 2: satu batch train dan test dari dataset.py."""
    from dataset import get_dataloaders  # import di sini agar error-nya terlokalisasi

    train_loader, test_loader = get_dataloaders(ctx["data_root"], ctx["batch_size"],
                                                ctx["num_workers"])
    try:
        L.log(f"  len(train_loader)={len(train_loader)} batch | "
              f"len(test_loader)={len(test_loader)} batch")
    except TypeError:
        L.log("  (loader tidak mendukung len())")

    for nama, loader in (("train", train_loader), ("test", test_loader)):
        batch = next(iter(loader))
        cek(isinstance(batch, dict), f"batch {nama} harus dict, didapat {type(batch).__name__}")
        cek({"A", "B"} <= set(batch.keys()),
            f"batch {nama} harus punya kunci 'A' dan 'B', didapat {list(batch.keys())}")
        _cek_gambar(batch["A"], f"{nama}['A']", ctx, L)  # Monet
        _cek_gambar(batch["B"], f"{nama}['B']", ctx, L)  # Foto
        cek(batch["A"].shape[0] == batch["B"].shape[0],
            f"batch {nama}: jumlah gambar A ({batch['A'].shape[0]}) != B ({batch['B'].shape[0]})")
        if nama == "train":
            ctx["batch"] = batch  # dipakai ulang di Langkah 3 dan 4


def langkah_model(ctx: dict, L: Logger):
    """Langkah 3: forward Generator dan Discriminator."""
    import torch
    from models import Generator, Discriminator

    device = _device(ctx)
    real_A, real_B, sumber = _ambil_batch(ctx, L, device)
    L.log(f"  Sumber input: {sumber}")

    G_AB, G_BA = Generator().to(device), Generator().to(device)
    D_A, D_B = Discriminator().to(device), Discriminator().to(device)
    for nama, m in (("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)):
        n = sum(p.numel() for p in m.parameters())
        L.log(f"  {nama:<5}: {n / 1e6:6.2f} juta parameter")

    _reset_vram(device)
    with torch.no_grad():  # hanya cek shape, tidak perlu menyimpan graf (hemat VRAM)
        fake_B = G_AB(real_A)  # Monet -> Foto
        fake_A = G_BA(real_B)  # Foto -> Monet
        _cek_gambar(fake_B, "G_AB(real_A)", ctx, L)
        _cek_gambar(fake_A, "G_BA(real_B)", ctx, L)

        bs = real_A.shape[0]
        uji_D = (("D_A(real_A)", D_A, real_A), ("D_A(fake_A)", D_A, fake_A),
                 ("D_B(real_B)", D_B, real_B), ("D_B(fake_B)", D_B, fake_B))
        for label, D, x in uji_D:
            p = D(x)
            L.log(f"  {label:<14} {_ringkas(p)}")
            cek(tuple(p.shape) == (bs, *D_OUT_SHAPE),
                f"{label}: output harus {(bs, *D_OUT_SHAPE)}, didapat {tuple(p.shape)}")
            cek(bool(torch.isfinite(p).all()), f"{label}: output mengandung NaN/Inf")
    _log_vram(device, L, "forward no_grad")

    ctx["models"] = (G_AB, G_BA, D_A, D_B)  # dipakai ulang di Langkah 4
    return "data dummy" if sumber == "dummy" else ""


def langkah_loss(ctx: dict, L: Logger):
    """Langkah 4: semua komponen loss + backward, cek gradien sampai ke 4 jaringan."""
    import torch
    from losses import CycleGANLoss

    device = _device(ctx)
    real_A, real_B, sumber = _ambil_batch(ctx, L, device)
    if "models" in ctx:
        G_AB, G_BA, D_A, D_B = ctx["models"]
    else:  # Langkah 3 gagal: coba bangun model baru agar loss tetap teruji
        from models import Generator, Discriminator
        G_AB, G_BA = Generator().to(device), Generator().to(device)
        D_A, D_B = Discriminator().to(device), Discriminator().to(device)
    crit = CycleGANLoss().to(device)

    try:
        _reset_vram(device)

        # --- Loss generator (urutan sama seperti satu iterasi training) ---
        fake_B = G_AB(real_A)   # Monet -> Foto
        fake_A = G_BA(real_B)   # Foto -> Monet
        rec_A = G_BA(fake_B)    # Monet -> Foto -> Monet
        rec_B = G_AB(fake_A)    # Foto -> Monet -> Foto
        idt_A = G_BA(real_A)    # G_BA keluarannya domain Monet, jadi input Monet harus tetap
        idt_B = G_AB(real_B)    # G_AB keluarannya domain Foto, jadi input Foto harus tetap

        adv_AB = crit.adversarial(D_B(fake_B), True)  # G_AB ingin menipu D_B
        adv_BA = crit.adversarial(D_A(fake_A), True)  # G_BA ingin menipu D_A
        cyc_A, cyc_A_raw = crit.cycle(real_A, rec_A)
        cyc_B, cyc_B_raw = crit.cycle(real_B, rec_B)
        idt_A_w, idt_A_raw = crit.identity(real_A, idt_A)
        idt_B_w, idt_B_raw = crit.identity(real_B, idt_B)

        L.log("  Loss generator:")
        for label, v in (("adversarial G_AB (D_B)", adv_AB), ("adversarial G_BA (D_A)", adv_BA),
                         ("cycle A (berbobot)", cyc_A), ("cycle B (berbobot)", cyc_B),
                         ("identity A (berbobot)", idt_A_w), ("identity B (berbobot)", idt_B_w)):
            _cek_skalar(v, label, L)
        for label, v in (("cycle A (mentah)", cyc_A_raw), ("cycle B (mentah)", cyc_B_raw),
                         ("identity A (mentah)", idt_A_raw), ("identity B (mentah)", idt_B_raw)):
            _cek_skalar(v, label, L, perlu_grad=False)

        # Info rasio bobot (tidak di-assert karena definisi lambda bisa berbeda)
        for label, w, r in (("cycle", cyc_A, cyc_A_raw), ("identity", idt_A_w, idt_A_raw)):
            if abs(r.item()) > 1e-12:  # .item(): ambil nilai tanpa memicu UserWarning autograd
                L.log(f"  rasio berbobot/mentah {label:<8} = {w.item() / r.item():.3f}")

        # AdversarialLoss harus peka terhadap target real/fake
        with torch.no_grad():
            p = D_A(real_A)
            v_real = float(crit.adversarial(p, True))
            v_fake = float(crit.adversarial(p, False))
        L.log(f"  adversarial(target real)={v_real:.4f} | (target fake)={v_fake:.4f}")
        if abs(v_real - v_fake) < 1e-8:
            peringatan(ctx, L, "AdversarialLoss memberi nilai sama untuk target real dan fake")

        loss_G = adv_AB + adv_BA + cyc_A + cyc_B + idt_A_w + idt_B_w
        _cek_skalar(loss_G, "TOTAL loss_G", L)
        loss_G.backward()
        _cek_gradien(G_AB, "G_AB", L)
        _cek_gradien(G_BA, "G_BA", L)

        # --- Loss diskriminator (fake di-detach agar gradien tidak ke generator) ---
        for m in (D_A, D_B):
            m.zero_grad(set_to_none=True)
        loss_D_A = crit.discriminator(D_A(real_A), D_A(fake_A.detach()))
        loss_D_B = crit.discriminator(D_B(real_B), D_B(fake_B.detach()))
        L.log("  Loss diskriminator:")
        _cek_skalar(loss_D_A, "loss_D_A", L)
        _cek_skalar(loss_D_B, "loss_D_B", L)
        (loss_D_A + loss_D_B).backward()
        _cek_gradien(D_A, "D_A", L)
        _cek_gradien(D_B, "D_B", L)

        _log_vram(device, L, "forward+backward fp32")
    finally:
        # Lepas semua tensor & model agar VRAM bebas untuk train.py di Langkah 5
        ctx.pop("models", None)
        ctx.pop("batch", None)
        del G_AB, G_BA, D_A, D_B, crit
        _bersihkan_vram()
    return "data dummy" if sumber == "dummy" else ""


def langkah_training(ctx: dict, L: Logger):
    """Langkah 5: train.py 1 epoch dengan --max_iters kecil, lewat subprocess."""
    if ctx["skip_train"]:
        raise LangkahDilewati("--skip_train aktif")

    nama_run = f"smoke_test_{ctx['stamp']}"
    cmd = [sys.executable, "train.py",
           "--epochs", "1",
           "--max_iters", str(ctx["max_iters"]),
           "--batch_size", str(ctx["batch_size"]),
           "--num_workers", str(ctx["num_workers"]),
           "--data_root", str(ctx["data_root"]),
           "--name", nama_run]
    if ctx["no_amp"]:
        cmd.append("--no_amp")
    L.log("  Perintah: " + subprocess.list2cmdline(cmd))
    L.log(f"  Batas waktu: {ctx['timeout']} s")
    L.log("  ---- output train.py ----")

    # Unbuffered + UTF-8 agar output muncul real-time dan aman di konsol Windows
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")

    habis_waktu = threading.Event()

    def _hentikan():
        habis_waktu.set()
        proc.kill()

    timer = threading.Timer(ctx["timeout"], _hentikan)
    timer.start()
    ada_nan = False
    try:
        for baris in proc.stdout:  # stream baris per baris ke layar + log
            baris = baris.rstrip("\r\n")
            if not baris.strip():
                continue
            L.log("  | " + baris)
            if RE_NAN.search(baris):
                ada_nan = True
        kode = proc.wait()
    finally:
        timer.cancel()
        proc.stdout.close()  # tutup pipa output secara eksplisit (hindari ResourceWarning)
    L.log("  ---- akhir output train.py ----")

    cek(not habis_waktu.is_set(),
        f"train.py melewati batas waktu {ctx['timeout']} s dan dihentikan paksa")
    cek(kode == 0, f"train.py keluar dengan kode {kode} (lihat traceback di output di atas)")
    if ada_nan:
        peringatan(ctx, L, "Output train.py mengandung 'nan'/'inf', cek nilai loss")
    return f"run --name {nama_run}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Smoke test integrasi CycleGAN Monet<->Foto")
    ap.add_argument("--data_root", type=Path, default=ROOT_DIR / "data" / "monet2photo")
    ap.add_argument("--batch_size", type=int, default=2,
                    help="default 2 agar bug dimensi batch tidak tersembunyi")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="default 0: paling aman di Windows dan traceback lebih jelas")
    ap.add_argument("--max_iters", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=600, help="batas waktu train.py (detik)")
    ap.add_argument("--no_amp", action="store_true", help="teruskan --no_amp ke train.py")
    ap.add_argument("--skip_train", action="store_true", help="lewati Langkah 5")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:  # konsol Windows: hindari crash karena karakter non-ASCII
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    L = Logger(ROOT_DIR / "logs" / f"smoke_test_{stamp}.txt")
    ctx = {
        "stamp": stamp,
        "data_root": args.data_root.resolve(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_iters": args.max_iters,
        "timeout": args.timeout,
        "no_amp": args.no_amp,
        "skip_train": args.skip_train,
        "peringatan": [],
    }

    L.log(f"SMOKE TEST CycleGAN Monet<->Foto | {stamp}")
    langkah = [
        ("Info lingkungan", langkah_lingkungan),
        ("Cek folder dataset", langkah_folder),
        ("Batch dari dataset.py (shape & rentang)", langkah_dataset),
        ("Forward Generator & Discriminator", langkah_model),
        ("Perhitungan loss + backward", langkah_loss),
        ("Training 1 epoch via train.py", langkah_training),
    ]
    hasil = [jalankan_langkah(i, nama, fn, ctx, L) for i, (nama, fn) in enumerate(langkah)]

    # ---- Ringkasan ----
    L.log("")
    L.log("=" * 72)
    L.log("RINGKASAN")
    L.log("=" * 72)
    L.log(f"{'No':<3} {'Langkah':<42} {'Status':<9} {'Durasi':>7}")
    L.log("-" * 72)
    for h in hasil:
        L.log(f"{h['no']:<3} {h['nama']:<42} {h['status']:<9} {h['durasi']:6.1f}s")
        if h["status"] != "OK" and h["catatan"]:
            L.log(f"    -> {h['catatan']}")
    L.log("-" * 72)

    n_ok = sum(h["status"] == "OK" for h in hasil)
    n_gagal = sum(h["status"] == "GAGAL" for h in hasil)
    n_skip = sum(h["status"] == "DILEWATI" for h in hasil)
    L.log(f"Hasil: {n_ok} OK, {n_gagal} GAGAL, {n_skip} DILEWATI dari {len(hasil)} langkah")
    if ctx["peringatan"]:
        L.log(f"Peringatan ({len(ctx['peringatan'])}):")
        for p in dict.fromkeys(ctx["peringatan"]):  # hilangkan duplikat, urutan tetap
            L.log(f"  - {p}")
    L.log(f"Log tersimpan di: {L.path}")
    L.close()
    return 1 if n_gagal else 0


if __name__ == "__main__":  # wajib di Windows bila DataLoader memakai num_workers > 0
    sys.exit(main())
