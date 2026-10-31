"""
plot_results.py
Membaca semua outputs/<eksperimen>/loss_log.csv lalu membuat:
  1. outputs/comparison_losses.png : 4 panel (G loss, D loss, cycle loss, identity loss)
     dengan satu garis per eksperimen.
  2. outputs/comparison_D_scores.png : skor rata-rata D untuk gambar asli vs palsu
     (alat bantu mendeteksi diskriminator yang terlalu dominan).
  3. outputs/summary_table.csv dan outputs/summary_table.md : tabel ringkas hasil akhir
     (rata-rata 5 epoch terakhir supaya tidak terpengaruh fluktuasi satu epoch)
     plus cycle reconstruction error dari evaluate.py jika sudah dijalankan.

Catatan: cycle loss dan identity loss yang digambar adalah nilai L1 MENTAH (tanpa
dikali lambda), jadi eksperimen lambda_cycle 5/10/15 bisa dibandingkan langsung.
G loss total tetap memuat bobot lambda, jadi nilainya wajar lebih besar pada lambda besar.

Contoh:
  python plot_results.py --output_dir outputs
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # tanpa layar, cukup simpan ke file
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Palet kategorikal tetap (urutan tidak diputar), sudah diuji aman untuk buta warna
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Gaya garis berbeda sebagai penanda kedua, supaya identitas tidak hanya dari warna
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS = [
    ("loss_G", "Generator loss (total, berbobot)"),
    ("loss_D", "Discriminator loss (D_A + D_B)"),
    ("loss_cycle", "Cycle loss (L1 mentah, A + B)"),
    ("loss_identity", "Identity loss (L1 mentah, A + B)"),
]


def sort_key(name):
    """Urutkan eksperimen secara stabil: baseline dulu, lalu alfabet. Warna mengikuti
    nama eksperimen, bukan urutan rank, jadi konsisten di semua grafik."""
    return (0 if name == "cyc10_id5_bs1" else 1, name)


def load_logs(output_dir):
    logs = {}
    for path in glob.glob(os.path.join(output_dir, "*", "loss_log.csv")):
        name = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path)
        if len(df):
            # Kalau ada epoch dobel (misal hasil resume), ambil catatan terakhir
            logs[name] = df.drop_duplicates("epoch", keep="last").sort_values("epoch")
    return dict(sorted(logs.items(), key=lambda kv: sort_key(kv[0])))


def style_axis(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left", pad=8)
    ax.set_xlabel("Epoch", fontsize=9, color=TEXT_2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # epoch selalu bilangan bulat
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_losses(logs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=SURFACE)
    names = list(logs)
    for ax, (col, title) in zip(axes.flat, PANELS):
        for i, name in enumerate(names):
            df = logs[name]
            if col in df:
                ax.plot(df["epoch"], df[col], color=PALETTE[i % len(PALETTE)],
                        linestyle=LINESTYLES[i % len(LINESTYLES)], linewidth=2, label=name)
        style_axis(ax, title, "Loss")
    # Satu legenda bersama di atas, berlaku untuk keempat panel
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(names), 5), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Perbandingan loss antar eksperimen CycleGAN (Monet ↔ Foto)",
                 fontsize=13, color=TEXT, y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_d_scores(logs, out_path):
    """Dua panel: rata-rata output D untuk gambar asli dan untuk gambar palsu.
    Target LSGAN: asli -> 1, palsu -> 0."""
    if not any("D_real_score" in df for df in logs.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    for i, (name, df) in enumerate(logs.items()):
        kw = dict(color=PALETTE[i % len(PALETTE)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                  linewidth=2, label=name)
        axes[0].plot(df["epoch"], df["D_real_score"], **kw)
        axes[1].plot(df["epoch"], df["D_fake_score"], **kw)
    style_axis(axes[0], "Skor D pada gambar ASLI (target 1)", "Rata-rata output D")
    style_axis(axes[1], "Skor D pada gambar PALSU (target 0)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(logs), 5), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def diagnose(df, last_n):
    """Catatan otomatis sederhana. Ini hanya PETUNJUK, tetap cek grid gambar."""
    tail = df.tail(last_n)
    notes = []
    if "loss_D" in df and tail["loss_D"].mean() < 0.05:
        notes.append("D sangat rendah: D mungkin terlalu dominan")
    if "D_real_score" in df and "D_fake_score" in df:
        gap = tail["D_real_score"].mean() - tail["D_fake_score"].mean()
        if gap > 0.8:
            notes.append("D hampir sempurna membedakan asli/palsu")
    if "loss_G_adv" in df and len(df) >= 2 * last_n:
        early = df.head(last_n)["loss_G_adv"].mean()
        if tail["loss_G_adv"].mean() > 1.5 * early:
            notes.append("G_adv terus naik: G kalah dari D")
    if "loss_cycle" in df and len(df) >= 2 * last_n:
        if tail["loss_cycle"].mean() > df.head(last_n)["loss_cycle"].mean():
            notes.append("cycle loss tidak turun")
    return "; ".join(notes) if notes else "normal"


def summary_table(logs, output_dir, last_n):
    rows = []
    for name, df in logs.items():
        tail = df.tail(last_n)
        row = {"experiment": name, "epochs": int(df["epoch"].max())}
        for col in ("loss_G", "loss_D", "loss_cycle", "loss_identity", "loss_G_adv",
                    "D_real_score", "D_fake_score"):
            if col in df:
                row[col] = round(tail[col].mean(), 4)
        if "epoch_time_s" in df:
            row["menit_per_epoch"] = round(df["epoch_time_s"].mean() / 60, 2)
        row["catatan"] = diagnose(df, last_n)
        rows.append(row)
    table = pd.DataFrame(rows)

    eval_path = os.path.join(output_dir, "eval_results.csv")
    if os.path.exists(eval_path) and len(table):
        ev = pd.read_csv(eval_path)[["experiment", "rec_err_A", "rec_err_B", "rec_err_mean"]]
        ev[["rec_err_A", "rec_err_B", "rec_err_mean"]] = ev[["rec_err_A", "rec_err_B", "rec_err_mean"]].round(4)
        table = table.merge(ev, on="experiment", how="left")
        # Pindahkan kolom catatan ke paling akhir
        table = table[[c for c in table.columns if c != "catatan"] + ["catatan"]]

    table.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)
    with open(os.path.join(output_dir, "summary_table.md"), "w") as f:
        f.write(f"Nilai loss = rata-rata {last_n} epoch terakhir. "
                "rec_err = cycle reconstruction error (L1) pada data test.\n\n")
        f.write(to_markdown(table))
    return table


def to_markdown(df):
    """Tabel markdown sederhana tanpa dependensi tambahan (tabulate)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Grafik & tabel perbandingan eksperimen")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--last_n", type=int, default=5, help="rata-rata N epoch terakhir untuk tabel")
    args = p.parse_args()

    logs = load_logs(args.output_dir)
    if not logs:
        print(f"[ERROR] Tidak ada loss_log.csv di {args.output_dir}/*/")
        return
    print(f"[INFO] Ditemukan {len(logs)} eksperimen: {', '.join(logs)}")

    plot_losses(logs, os.path.join(args.output_dir, "comparison_losses.png"))
    has_scores = plot_d_scores(logs, os.path.join(args.output_dir, "comparison_D_scores.png"))
    table = summary_table(logs, args.output_dir, args.last_n)

    print(f"[INFO] Grafik: {args.output_dir}/comparison_losses.png"
          + (f" dan {args.output_dir}/comparison_D_scores.png" if has_scores else ""))
    print(f"[INFO] Tabel : {args.output_dir}/summary_table.csv dan summary_table.md\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
