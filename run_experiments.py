"""
run_experiments.py
Menjalankan eksperimen tuning secara berurutan, lalu mengevaluasi dan membuat grafik.

Desain eksperimen (one-factor-at-a-time terhadap baseline cyc10_id5_bs1):
  (1) lambda_cycle   : 5, 10, 15          (identity 5, batch 1)
  (2) identity loss  : 0 vs 5             (cycle 10, batch 1)
  (3) batch_size     : 1 vs 2             (cycle 10, identity 5)
Baseline dipakai bersama oleh ketiga kelompok, sehingga totalnya cukup 5 run:
  cyc5_id5_bs1, cyc10_id5_bs1 (baseline), cyc15_id5_bs1, cyc10_id0_bs1, cyc10_id5_bs2

Fitur:
- Eksperimen yang sudah selesai (ada file DONE) dilewati, jadi aman dijalankan ulang.
- Eksperimen yang terputus dilanjutkan otomatis (--resume).
- Satu eksperimen gagal tidak menghentikan yang lain.
- Output terminal tiap eksperimen juga disimpan ke outputs/<nama>/train_stdout.log.

Contoh:
  python run_experiments.py --data_root data/monet2photo --epochs 40
  python run_experiments.py --only cyc10_id5_bs1 cyc10_id0_bs1   # sebagian saja
  python run_experiments.py --dry_run                            # lihat perintahnya saja
"""

import argparse
import os
import subprocess
import sys
import time

BASE = {"lambda_cycle": 10, "lambda_identity": 5, "batch_size": 1}


def build_configs():
    """Susun daftar konfigurasi unik dari ketiga kelompok eksperimen."""
    groups = {
        "lambda_cycle": [dict(BASE, lambda_cycle=v) for v in (5, 10, 15)],
        "identity": [dict(BASE, lambda_identity=v) for v in (0, 5)],
        "batch_size": [dict(BASE, batch_size=v) for v in (1, 2)],
    }
    configs, seen = [], set()
    for group_name, cfgs in groups.items():
        for c in cfgs:
            name = f"cyc{c['lambda_cycle']:g}_id{c['lambda_identity']:g}_bs{c['batch_size']}"
            if name not in seen:
                seen.add(name)
                configs.append(dict(c, name=name))
    return configs


def run(cmd, log_path=None):
    """Jalankan perintah dan tampilkan outputnya di terminal (sekaligus simpan ke log)."""
    print("$ " + " ".join(cmd), flush=True)
    log_f = open(log_path, "a") if log_path else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        if log_f:
            log_f.write(line)
    proc.wait()
    if log_f:
        log_f.close()
    return proc.returncode


def main():
    p = argparse.ArgumentParser(description="Jalankan seluruh eksperimen tuning CycleGAN")
    p.add_argument("--data_root", type=str, default="data/monet2photo")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--only", nargs="*", default=None, help="jalankan hanya eksperimen dengan nama ini")
    p.add_argument("--max_iters", type=int, default=0, help="untuk uji cepat; 0 = penuh")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="argumen tambahan untuk train.py, taruh paling akhir")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    configs = build_configs()
    if args.only:
        configs = [c for c in configs if c["name"] in args.only]

    print("Daftar eksperimen:")
    for c in configs:
        print(f"  - {c['name']}: lambda_cycle={c['lambda_cycle']}, "
              f"lambda_identity={c['lambda_identity']}, batch_size={c['batch_size']}")

    status = {}
    for c in configs:
        exp_dir = os.path.join(args.output_dir, c["name"])
        if os.path.exists(os.path.join(exp_dir, "DONE")):
            print(f"\n[LEWATI] {c['name']} sudah selesai.")
            status[c["name"]] = "sudah selesai"
            continue
        cmd = [py, os.path.join(here, "train.py"),
               "--data_root", args.data_root, "--output_dir", args.output_dir,
               "--epochs", str(args.epochs), "--batch_size", str(c["batch_size"]),
               "--lambda_cycle", str(c["lambda_cycle"]), "--lambda_identity", str(c["lambda_identity"]),
               "--name", c["name"], "--num_workers", str(args.num_workers),
               "--save_every", str(args.save_every), "--resume"]
        if args.max_iters:
            cmd += ["--max_iters", str(args.max_iters)]
        cmd += args.extra
        print(f"\n===== Eksperimen {c['name']} =====")
        if args.dry_run:
            print("$ " + " ".join(cmd))
            continue
        os.makedirs(exp_dir, exist_ok=True)
        t0 = time.time()
        code = run(cmd, os.path.join(exp_dir, "train_stdout.log"))
        mins = (time.time() - t0) / 60
        status[c["name"]] = f"OK ({mins:.1f} menit)" if code == 0 else f"GAGAL (exit {code}), lihat train_stdout.log"

    if args.dry_run:
        return

    print("\n===== Ringkasan training =====")
    for k, v in status.items():
        print(f"  {k}: {v}")

    names = [c["name"] for c in configs]
    if not args.skip_eval and names:
        print("\n===== Evaluasi data test =====")
        run([py, os.path.join(here, "evaluate.py"), "--data_root", args.data_root,
             "--output_dir", args.output_dir, "--num_workers", str(args.num_workers),
             "--experiments", *names])
    print("\n===== Grafik perbandingan =====")
    run([py, os.path.join(here, "plot_results.py"), "--output_dir", args.output_dir])


if __name__ == "__main__":
    main()
