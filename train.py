"""
train.py
Training loop CycleGAN (Monet <-> Foto, 128x128) yang hemat VRAM untuk RTX 3050.

Urutan update dalam SATU iterasi:
  1. Update kedua GENERATOR (G_AB dan G_BA) sekaligus.
     - D_A dan D_B dibekukan (requires_grad=False) supaya gradien loss generator
       tidak ikut mengubah bobot diskriminator dan tidak membuang memori.
     - loss_G = adversarial(A->B) + adversarial(B->A) + cycle(A) + cycle(B) + identity(A) + identity(B)
  2. Update kedua DISKRIMINATOR (D_A dan D_B).
     - Memakai gambar palsu yang sudah di-.detach() dan diambil lewat ImageBuffer,
       sehingga gradien tidak mengalir balik ke generator.
     - loss_D = 0.5 * (MSE(D(real), 1) + MSE(D(fake), 0))

Kenapa generator dulu? Gambar palsu yang dibuat saat langkah generator langsung bisa
dipakai ulang (detach) untuk melatih diskriminator, jadi tidak perlu forward ulang
generator. Ini menghemat waktu dan VRAM. D lalu belajar dari keluaran G yang terbaru.

Contoh menjalankan:
  python train.py --data_root data/monet2photo --epochs 50 --batch_size 1 --name cyc10_id5_bs1
"""

import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import LambdaLR

from dataset import get_dataloaders
from losses import CycleGANLoss
from models import Generator, Discriminator
from utils import (CSVLogger, ImageBuffer, count_params, fixed_test_samples, linear_decay_lambda,
                   load_checkpoint, pick_device, save_checkpoint, save_sample_grid, set_seed)

# Kolom CSV log. loss_cycle dan loss_identity dicatat MENTAH (L1 tanpa lambda)
# supaya eksperimen dengan lambda berbeda bisa dibandingkan dengan adil.
LOG_FIELDS = [
    "epoch", "loss_G", "loss_D", "loss_cycle", "loss_identity",
    "loss_G_adv", "loss_cycle_weighted", "loss_identity_weighted",
    "loss_D_A", "loss_D_B", "D_real_score", "D_fake_score", "lr", "epoch_time_s",
]


def parse_args():
    p = argparse.ArgumentParser(description="Training CycleGAN Monet <-> Foto")
    p.add_argument("--data_root", type=str, default="data/monet2photo",
                   help="folder berisi trainA, trainB, testA, testB")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_identity", type=float, default=5.0)
    p.add_argument("--name", type=str, default="cyc10_id5_bs1", help="nama eksperimen")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=5, help="simpan checkpoint & grid tiap N epoch")
    p.add_argument("--buffer_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="lanjutkan dari outputs/<name>/checkpoint_latest.pth")
    p.add_argument("--no_amp", action="store_true", help="matikan mixed precision")
    p.add_argument("--print_every", type=int, default=100, help="cetak progres tiap N iterasi")
    p.add_argument("--max_iters", type=int, default=0,
                   help="batasi iterasi per epoch (untuk uji cepat). 0 = semua data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def unpack_batch(batch):
    """dataset.py mengembalikan dict {'A': Monet, 'B': foto}.
    Bentuk tuple (real_A, real_B) juga didukung untuk berjaga-jaga."""
    if isinstance(batch, dict):
        return batch["A"], batch["B"]
    return batch[0], batch[1]


def set_requires_grad(models, flag: bool):
    for m in models:
        for p in m.parameters():
            p.requires_grad = flag


def make_scaler(enabled: bool):
    """GradScaler untuk mixed precision. torch.amp adalah nama baru dari torch.cuda.amp
    (fungsinya sama, torch.cuda.amp.GradScaler sudah deprecated di PyTorch >= 2.4)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = pick_device()
    use_amp = (device.type == "cuda") and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # ukuran input tetap, cari algoritma conv tercepat

    exp_dir = os.path.join(args.output_dir, args.name)
    sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[INFO] Device: {device} | AMP: {use_amp} | Eksperimen: {args.name}")

    # ---------------- Data ----------------
    train_loader, test_loader = get_dataloaders(args.data_root, args.batch_size, args.num_workers)
    print(f"[INFO] Iterasi per epoch: {len(train_loader)}")
    # Ambil satu batch test tetap (fixed) untuk grid contoh, supaya progres antar epoch mudah dibandingkan
    fixed_A, fixed_B = fixed_test_samples(test_loader, 4)
    fixed_A, fixed_B = fixed_A.to(device), fixed_B.to(device)

    # ---------------- Model ----------------
    # Bobot sudah diinisialisasi N(0, 0.02) di dalam konstruktor (models.py)
    G_AB = Generator().to(device)  # Monet -> Foto
    G_BA = Generator().to(device)  # Foto  -> Monet
    D_A = Discriminator().to(device)  # menilai lukisan Monet
    D_B = Discriminator().to(device)  # menilai foto
    print(f"[INFO] Parameter: G={count_params(G_AB):,} per generator | D={count_params(D_A):,} per diskriminator")

    # ---------------- Loss, optimizer, scheduler ----------------
    criterion = CycleGANLoss(args.lambda_cycle, args.lambda_identity)
    # Satu optimizer untuk kedua generator, satu untuk kedua diskriminator
    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                             lr=args.lr, betas=(args.beta1, 0.999))
    sched_G = LambdaLR(opt_G, lr_lambda=linear_decay_lambda(args.epochs))
    sched_D = LambdaLR(opt_D, lr_lambda=linear_decay_lambda(args.epochs))
    # Scaler terpisah untuk G dan D karena loss-nya di-backward terpisah
    scaler_G = make_scaler(use_amp)
    scaler_D = make_scaler(use_amp)

    buffer_A = ImageBuffer(args.buffer_size)  # gambar Monet palsu (fake_A)
    buffer_B = ImageBuffer(args.buffer_size)  # gambar foto palsu (fake_B)

    models = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}
    optimizers = {"G": opt_G, "D": opt_D}
    schedulers = {"G": sched_G, "D": sched_D}
    scalers = {"G": scaler_G, "D": scaler_D}

    start_epoch = 0
    ckpt_latest = os.path.join(exp_dir, "checkpoint_latest.pth")
    if args.resume and os.path.exists(ckpt_latest):
        # State scheduler sengaja TIDAK dimuat: jadwal lr dihitung ulang dari --epochs
        # yang sekarang, jadi resume tetap benar walaupun jumlah epoch diubah
        ckpt = load_checkpoint(ckpt_latest, models, optimizers, None, scalers, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        lr_factor = linear_decay_lambda(args.epochs)(start_epoch)
        for opt, sched in ((opt_G, sched_G), (opt_D, sched_D)):
            for g in opt.param_groups:
                g["initial_lr"] = args.lr
                g["lr"] = args.lr * lr_factor
            sched.base_lrs = [args.lr] * len(opt.param_groups)
            sched.last_epoch = start_epoch
        # Seed digeser supaya urutan data acak tidak mengulang dari epoch 1
        set_seed(args.seed + start_epoch)
        print(f"[INFO] Melanjutkan dari epoch {start_epoch + 1} (lr {args.lr * lr_factor:.2e})")
    logger = CSVLogger(os.path.join(exp_dir, "loss_log.csv"), LOG_FIELDS, resume=args.resume and start_epoch > 0)

    use_identity = args.lambda_identity > 0

    # ---------------- Loop epoch ----------------
    for epoch in range(start_epoch, args.epochs):
        for m in models.values():
            m.train()
        t0 = time.time()
        sums = {k: 0.0 for k in LOG_FIELDS if k.startswith(("loss", "D_"))}
        n_iter = 0

        for i, batch in enumerate(train_loader):
            if args.max_iters and i >= args.max_iters:
                break
            real_A, real_B = unpack_batch(batch)
            real_A = real_A.to(device, non_blocking=True)
            real_B = real_B.to(device, non_blocking=True)

            # ===================== 1. Update generator =====================
            set_requires_grad([D_A, D_B], False)  # bekukan D
            opt_G.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp):
                fake_B = G_AB(real_A)   # Monet -> foto palsu
                fake_A = G_BA(real_B)   # foto -> Monet palsu
                rec_A = G_BA(fake_B)    # Monet -> foto -> Monet
                rec_B = G_AB(fake_A)    # foto -> Monet -> foto

                # Adversarial: G ingin D menganggap gambar palsu sebagai asli (target 1)
                loss_adv_AB = criterion.adversarial(D_B(fake_B), True)
                loss_adv_BA = criterion.adversarial(D_A(fake_A), True)
                loss_G_adv = loss_adv_AB + loss_adv_BA

                # Cycle consistency
                cyc_A_w, cyc_A_raw = criterion.cycle(real_A, rec_A)
                cyc_B_w, cyc_B_raw = criterion.cycle(real_B, rec_B)
                loss_cycle_w = cyc_A_w + cyc_B_w
                loss_cycle_raw = cyc_A_raw + cyc_B_raw

                # Identity: G_AB menghasilkan foto, jadi input foto harus tetap sama
                if use_identity:
                    id_B_w, id_B_raw = criterion.identity(real_B, G_AB(real_B))
                    id_A_w, id_A_raw = criterion.identity(real_A, G_BA(real_A))
                    loss_id_w = id_A_w + id_B_w
                    loss_id_raw = id_A_raw + id_B_raw
                else:
                    loss_id_w = torch.zeros((), device=device)

                loss_G = loss_G_adv + loss_cycle_w + loss_id_w

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # Jika identity loss dimatikan, tetap hitung nilai mentahnya TANPA gradien
            # hanya untuk logging (supaya grafik identity bisa dibandingkan antar eksperimen)
            if not use_identity:
                with torch.no_grad(), autocast_ctx(use_amp):
                    loss_id_raw = (criterion.l1(G_AB(real_B).float(), real_B.float())
                                   + criterion.l1(G_BA(real_A).float(), real_A.float()))

            # ===================== 2. Update diskriminator =====================
            set_requires_grad([D_A, D_B], True)
            opt_D.zero_grad(set_to_none=True)
            # Ambil gambar palsu dari buffer (sudah di-detach di dalam buffer)
            fake_A_buf = buffer_A.push_and_pop(fake_A)
            fake_B_buf = buffer_B.push_and_pop(fake_B)
            with autocast_ctx(use_amp):
                pred_real_A = D_A(real_A)
                pred_fake_A = D_A(fake_A_buf)
                loss_D_A = criterion.discriminator(pred_real_A, pred_fake_A)
                pred_real_B = D_B(real_B)
                pred_fake_B = D_B(fake_B_buf)
                loss_D_B = criterion.discriminator(pred_real_B, pred_fake_B)
                loss_D = loss_D_A + loss_D_B

            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # ===================== Catat =====================
            # Skor rata-rata D: dekat 1 untuk real, dekat 0 untuk fake berarti D sangat yakin
            d_real = 0.5 * (pred_real_A.float().mean() + pred_real_B.float().mean())
            d_fake = 0.5 * (pred_fake_A.float().mean() + pred_fake_B.float().mean())
            vals = {
                "loss_G": loss_G, "loss_D": loss_D, "loss_cycle": loss_cycle_raw,
                "loss_identity": loss_id_raw, "loss_G_adv": loss_G_adv,
                "loss_cycle_weighted": loss_cycle_w, "loss_identity_weighted": loss_id_w,
                "loss_D_A": loss_D_A, "loss_D_B": loss_D_B,
                "D_real_score": d_real, "D_fake_score": d_fake,
            }
            # Ubah ke angka Python biasa (lepas dari graf komputasi)
            vals = {k: float(v.detach()) for k, v in vals.items()}
            for k, v in vals.items():
                sums[k] += v
            n_iter += 1

            if args.print_every and (i + 1) % args.print_every == 0:
                print(f"  epoch {epoch + 1}/{args.epochs} iter {i + 1}/{len(train_loader)} "
                      f"| G {vals['loss_G']:.3f} | D {vals['loss_D']:.3f} "
                      f"| cyc {vals['loss_cycle']:.3f} | id {vals['loss_identity']:.3f}")

        # ---------------- Akhir epoch ----------------
        lr_now = opt_G.param_groups[0]["lr"]  # lr yang dipakai selama epoch ini
        sched_G.step()
        sched_D.step()
        elapsed = time.time() - t0
        row = {k: round(v / max(1, n_iter), 6) for k, v in sums.items()}
        row.update({"epoch": epoch + 1, "lr": lr_now, "epoch_time_s": round(elapsed, 1)})
        logger.log(row)
        print(f"[EPOCH {epoch + 1}/{args.epochs}] G {row['loss_G']:.4f} | D {row['loss_D']:.4f} "
              f"| cycle {row['loss_cycle']:.4f} | identity {row['loss_identity']:.4f} "
              f"| lr {lr_now:.2e} | {elapsed:.0f}s")
        if device.type == "cuda" and epoch == start_epoch:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"[INFO] Puncak VRAM terpakai: {peak:.2f} GB")

        # Checkpoint "latest" disimpan tiap epoch (untuk resume jika laptop mati),
        # checkpoint bernomor + grid contoh disimpan tiap save_every epoch dan di epoch terakhir
        save_checkpoint(ckpt_latest, epoch, models, optimizers, schedulers, scalers, extra=vars(args))
        is_last = (epoch + 1) == args.epochs
        if (epoch + 1) % args.save_every == 0 or is_last:
            for m in models.values():
                m.eval()
            with torch.no_grad(), autocast_ctx(use_amp):
                s_fake_B = G_AB(fixed_A)
                s_fake_A = G_BA(fixed_B)
                s_rec_A = G_BA(s_fake_B)
                s_rec_B = G_AB(s_fake_A)
            save_sample_grid(os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"),
                             fixed_A, s_fake_B, s_rec_A, fixed_B, s_fake_A, s_rec_B)
            # Checkpoint bernomor hanya berisi bobot generator & diskriminator (lebih kecil)
            torch.save({"epoch": epoch, "models": {k: m.state_dict() for k, m in models.items()}},
                       os.path.join(exp_dir, f"checkpoint_epoch_{epoch + 1:03d}.pth"))
            print(f"[INFO] Checkpoint & grid contoh disimpan di {exp_dir}")

    # Penanda bahwa eksperimen selesai (dipakai run_experiments.py untuk melewati yang sudah beres)
    with open(os.path.join(exp_dir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[SELESAI] Hasil ada di {exp_dir}")


if __name__ == "__main__":
    main()
