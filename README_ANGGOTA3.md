# Bagian Anggota 3: Training Loop & Hyperparameter Tuning (Zara)

File yang dikerjakan bagian ini:

| File | Isi |
| --- | --- |
| `losses.py` | Adversarial loss LSGAN (MSE), cycle-consistency loss (L1), identity loss (L1). `lambda_cycle` (default 10) dan `lambda_identity` (default 5) bisa diatur. |
| `utils.py` | `ImageBuffer` (50 gambar), simpan/muat checkpoint, grid contoh gambar 6 kolom, `CSVLogger`, jadwal lr linear decay, `set_seed`. |
| `train.py` | Training loop lengkap: argumen CLI, 2 optimizer Adam (lr 2e-4, beta1 0.5), scheduler, mixed precision, checkpoint & grid tiap 5 epoch, log CSV per epoch, resume. |
| `evaluate.py` | Cycle reconstruction error (rata-rata L1 asli vs rekonstruksi) pada data test. |
| `run_experiments.py` | Menjalankan 5 konfigurasi tuning berurutan, lalu evaluasi dan grafik. |
| `plot_results.py` | Grafik perbandingan 4 loss + skor diskriminator, plus tabel ringkas. |

Semua file ditaruh di folder yang sama dengan `models.py` (Anggota 1) dan `dataset.py` (Anggota 2).

## Antarmuka yang dipakai dari modul lain

Sudah dicocokkan dengan isi repo `Revanzavz/CycleGAN`:

- `models.py` (Vanza): `Generator()` dan `Discriminator()` dibuat tanpa argumen, input/output `(B, 3, 128, 128)`, D mengeluarkan peta `(B, 1, 14, 14)` tanpa sigmoid. Bobot sudah diinisialisasi N(0, 0.02) di konstruktor, jadi `train.py` tidak menginisialisasi ulang.
- `dataset.py` (Raihan): `get_dataloaders(root, batch_size, num_workers)` mengembalikan `(train_loader, test_loader)`, dan tiap batch berupa dict `{"A": Monet, "B": foto}` dalam rentang [-1, 1]. Folder default `data/monet2photo` (dibuat oleh `download_data.py`).

## Cara menjalankan

Perangkat dipilih otomatis: GPU NVIDIA (CUDA, dengan mixed precision) → GPU Apple Silicon (MPS) → CPU. Training sungguhan dijalankan di laptop RTX 3050; di MacBook cukup untuk uji cepat. Di Mac gunakan `python3`, atau aktifkan virtual environment dulu (`python3 -m venv .venv` lalu `source .venv/bin/activate`).

```bash
pip install torch torchvision pandas matplotlib

# 1. Uji cepat dulu (2 epoch, 20 iterasi per epoch) untuk memastikan semua modul nyambung
python train.py --data_root data/monet2photo --epochs 2 --max_iters 20 --save_every 1 --name uji_cepat

# 2. Training dasar (baseline)
python train.py --data_root data/monet2photo --epochs 50 --batch_size 1 --name cyc10_id5_bs1

# 3. Lanjutkan training yang terputus (laptop mati, dsb.)
python train.py --data_root data/monet2photo --epochs 50 --name cyc10_id5_bs1 --resume

# 4. Semua eksperimen tuning + evaluasi + grafik sekaligus
python run_experiments.py --data_root data/monet2photo --epochs 40

# 5. Kalau hanya ingin evaluasi / grafik ulang
python evaluate.py --data_root data/monet2photo --all
python plot_results.py
```

Setelah uji cepat, hapus folder `outputs/uji_cepat` supaya tidak ikut masuk grafik perbandingan.

Hasil tiap eksperimen ada di `outputs/<nama_eksperimen>/`:

```
config.json               argumen yang dipakai
loss_log.csv              loss per epoch
samples/epoch_005.png ... grid contoh (real A | fake B | rec A | real B | fake A | rec B)
checkpoint_latest.pth     checkpoint lengkap untuk resume (ditimpa tiap epoch)
checkpoint_epoch_005.pth  bobot model tiap 5 epoch
test_samples.png          hasil pada data test (dari evaluate.py)
DONE                      penanda eksperimen selesai
```

Grafik dan tabel gabungan: `outputs/comparison_losses.png`, `outputs/comparison_D_scores.png`, `outputs/summary_table.csv`, `outputs/summary_table.md`, `outputs/eval_results.csv`.

## Urutan update dalam satu iterasi (dan alasannya)

1. **Generator dulu.** D_A dan D_B dibekukan (`requires_grad=False`). Hitung `fake_B = G_AB(real_A)`, `fake_A = G_BA(real_B)`, rekonstruksi `rec_A = G_BA(fake_B)` dan `rec_B = G_AB(fake_A)`, lalu identity `G_AB(real_B)` dan `G_BA(real_A)`. Loss G = adversarial (G ingin D_B dan D_A menilai gambar palsu sebagai asli) + lambda_cycle × cycle + lambda_identity × identity. Kedua generator di-update bersama dengan satu optimizer karena loss cycle menghubungkan keduanya.
2. **Diskriminator kemudian.** Gambar palsu dari langkah 1 di-`detach()` dan dilewatkan `ImageBuffer`, lalu D_A dan D_B dilatih: asli → 1, palsu → 0, dikali 0.5.

Alasannya: gambar palsu dari langkah generator bisa langsung dipakai ulang untuk D tanpa forward ulang (hemat waktu dan VRAM). Membekukan D saat update G mencegah gradien tidak berguna mengalir ke D dan menghemat memori. `detach()` di langkah D memastikan loss D tidak mengubah generator.

## Keputusan desain lain

- **Mixed precision** (`torch.autocast` + `GradScaler`) mengurangi pemakaian VRAM dan mempercepat training di RTX 3050. Ini API baru dari `torch.cuda.amp` yang sudah deprecated; fungsinya sama. Loss dihitung dalam float32 supaya tidak overflow. Dua scaler dipakai (G dan D) karena backward-nya terpisah. Bisa dimatikan dengan `--no_amp` jika muncul NaN.
- **Log loss mentah.** `loss_cycle` dan `loss_identity` di CSV adalah L1 tanpa dikali lambda, supaya eksperimen lambda 5/10/15 bisa dibandingkan adil. Versi berbobotnya juga dicatat (`*_weighted`). Saat `lambda_identity = 0`, identity mentah tetap dihitung tanpa gradien hanya untuk logging.
- **Scheduler**: lr konstan pada separuh pertama epoch, turun linear ke 0 pada separuh kedua (faktor dicatat di kolom `lr`).
- **Checkpoint aman**: ditulis ke file sementara lalu diganti, jadi file lama tidak rusak bila laptop mati saat menyimpan.
- **Grid contoh memakai 4 gambar test yang sama tiap kali** (tidak bergantung batch size), sehingga perkembangan antar epoch bisa dibandingkan langsung.
- **Resume**: jadwal lr dihitung ulang dari `--epochs` saat ini (jadi jumlah epoch boleh ditambah saat melanjutkan), dan seed acak digeser supaya urutan data tidak mengulang dari epoch 1.

## Desain eksperimen tuning

One-factor-at-a-time terhadap baseline `cyc10_id5_bs1`. Baseline dipakai bersama sehingga cukup 5 run:

| Nama | lambda_cycle | lambda_identity | batch_size | Menguji |
| --- | --- | --- | --- | --- |
| cyc10_id5_bs1 | 10 | 5 | 1 | baseline |
| cyc5_id5_bs1 | 5 | 5 | 1 | lambda_cycle kecil |
| cyc15_id5_bs1 | 15 | 5 | 1 | lambda_cycle besar |
| cyc10_id0_bs1 | 10 | 0 | 1 | tanpa identity loss |
| cyc10_id5_bs2 | 10 | 5 | 2 | batch size 2 |

Perkiraan waktu: dengan ~1.070 lukisan Monet per epoch dan resolusi 128, satu epoch di RTX 3050 kira-kira 1 sampai 3 menit (cek kolom `epoch_time_s` di epoch pertama untuk angka pastinya). Lima run × 40 epoch bisa memakan beberapa jam, jadi jalankan semalaman atau bagi per hari dengan `--only`. Script aman dihentikan dan dijalankan ulang: yang sudah selesai dilewati, yang terputus dilanjutkan.

Catatan untuk analisis batch size: dengan jumlah epoch sama, batch 2 berarti jumlah update bobot hanya separuh batch 1. Sebutkan ini di laporan saat membandingkan hasilnya.

## Cara membaca grafik

`comparison_losses.png` berisi 4 panel (G loss, D loss, cycle loss, identity loss), dan `comparison_D_scores.png` menampilkan rata-rata output D untuk gambar asli (target 1) dan palsu (target 0).

**Training sehat** biasanya terlihat seperti ini: D loss berada di kisaran 0,1 sampai 0,3 dan naik-turun tanpa menuju nol; adversarial G stabil di sekitar 0,3 sampai 1; cycle loss turun cepat di awal lalu melandai; skor D asli dan palsu tidak terpisah jauh (misalnya sekitar 0,6 vs 0,3), tanda D dan G saling mengimbangi.

**Diskriminator terlalu dominan:**
- D loss turun mendekati 0 dan bertahan di sana.
- Skor D untuk gambar asli mendekati 1 dan untuk gambar palsu mendekati 0 (jarak keduanya > 0,8).
- Komponen adversarial G (`loss_G_adv` di CSV) terus naik.
- Akibatnya G hampir tidak mendapat sinyal berguna; grid gambar kabur atau berisi artefak.
- Solusi yang bisa dicoba: turunkan lr diskriminator, perbesar buffer, atau naikkan lambda_cycle.

**Mode collapse** (G menghasilkan keluaran yang hampir sama untuk input berbeda):
- Grafik loss **tidak bisa mendeteksinya sendirian**, jadi selalu cek grid contoh gambar. Tanda utamanya: di grid, kolom `fake B` atau `fake A` terlihat mirip untuk semua baris walaupun inputnya berbeda.
- Petunjuk di grafik: adversarial G tiba-tiba turun sangat rendah sementara D loss naik (G menemukan satu pola yang "menipu" D), lalu loss berosilasi besar.
- Pada CycleGAN, mode collapse penuh jarang terjadi karena cycle loss memaksa gambar bisa dikembalikan ke aslinya. Kalau cycle loss **naik** sementara adversarial G turun, itu tanda G mulai mengorbankan isi gambar demi menipu D.

**Membandingkan eksperimen:**
- lambda_cycle besar → cycle loss mentah lebih kecil dan rekonstruksi bagus, tapi perubahan gaya bisa lebih "malu-malu". lambda_cycle kecil → gaya lebih berani, tapi isi gambar bisa berubah.
- Tanpa identity loss (`id0`) → identity mentah biasanya lebih tinggi dan warna hasil bisa bergeser (misalnya langit berubah warna).
- Kolom `catatan` di `summary_table.md` memberi peringatan otomatis sederhana (D dominan, G_adv naik, cycle tidak turun). Itu hanya petunjuk, bukan kesimpulan.
- `rec_err_mean` (dari `evaluate.py`) makin kecil makin konsisten. Tapi error yang sangat kecil bisa juga berarti generator hampir tidak mengubah gambar, jadi baca bersama grid `test_samples.png`.

## Draf Log Interaksi (untuk Dafa)

```
LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]
• Tanggal Sesi: 31/10/2026
• Target Modul: losses.py, utils.py, train.py, evaluate.py, run_experiments.py, plot_results.py
• Tautan Sesi Chat: [share link sesi ini]

1. TAHAP INISIASI
   - Prompt Kelompok: (Prompt 1 dan Prompt 2 Anggota 3 dari dokumen prompting)
   - Alasan Khusus: Training dipecah menjadi tiga tahap (loss, utils, training loop)
     agar tiap modul bisa diuji terpisah sebelum digabung. Antarmuka models.py dan
     dataset.py disebutkan eksplisit supaya kode tidak bentrok dengan anggota lain.
     Mixed precision diminta karena VRAM RTX 3050 terbatas.
   - Output Model: losses.py (LSGAN + L1 cycle + L1 identity berbobot), utils.py
     (ImageBuffer 50, checkpoint, grid 6 kolom, CSV logger), train.py (update G lalu D,
     Adam 2e-4 beta1 0.5, linear decay, AMP, checkpoint tiap 5 epoch), serta skrip
     eksperimen, evaluasi, dan grafik.

2. TAHAP PENANGANAN EROR / DEBUGGING
   - [isi saat training sungguhan dijalankan dan muncul error]

3. TAHAP OPTIMASI
   - Eksperimen: lambda_cycle 5/10/15, identity 0 vs 5, batch 1 vs 2 (5 run).
   - [isi tabel summary_table.md dan analisis setelah eksperimen selesai]
```
