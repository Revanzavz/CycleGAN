"""
download_data.py
Mengunduh dan menyiapkan dataset monet2photo untuk proyek CycleGAN.

Hasil akhir (default, di samping file ini):
    data/monet2photo/trainA   -> lukisan Monet (domain A)
    data/monet2photo/trainB   -> foto pemandangan (domain B)
    data/monet2photo/testA    -> lukisan Monet untuk pengujian
    data/monet2photo/testB    -> foto pemandangan untuk pengujian

Pemakaian:
    python download_data.py                       # unduh, ekstrak, verifikasi
    python download_data.py --verify-only         # hanya cek folder yang sudah ada
    python download_data.py --zip-path file.zip   # pakai zip hasil unduhan manual
    python download_data.py --force               # unduh ulang dan timpa folder lama

Kode keluar: 0 jika dataset siap, 1 jika gagal (pesan penyebab dicetak ke layar).
Hanya memakai library standar Python (tanpa dependensi tambahan).
"""

import argparse
import errno
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
NAMA_DATASET = "monet2photo"
SUBFOLDER = ("trainA", "trainB", "testA", "testB")
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png"}

# Keterangan untuk tabel keluaran, mengikuti konvensi proyek: A = Monet, B = foto
KETERANGAN = {
    "trainA": "Monet (domain A), training",
    "trainB": "Foto (domain B), training",
    "testA": "Monet (domain A), pengujian",
    "testB": "Foto (domain B), pengujian",
}

# Dicoba berurutan: alamat kedua dipakai jika alamat pertama gagal
URL_BAWAAN = (
    "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip",
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip",
)

JUMLAH_PERCOBAAN = 3      # percobaan per alamat
TIMEOUT_DETIK = 30        # batas tunggu koneksi/baca data
UKURAN_CHUNK = 256 * 1024  # baca per 256 KB agar progres terlihat

# Folder data default diletakkan di samping skrip ini, jadi hasilnya sama
# siapa pun yang menjalankan dan dari folder mana pun skrip dipanggil
DIR_BAWAAN = Path(__file__).resolve().parent / "data"


class DataError(Exception):
    """Kesalahan yang bisa dijelaskan ke pengguna; pesannya siap ditampilkan."""


# ---------------------------------------------------------------------------
# Verifikasi
# ---------------------------------------------------------------------------
def hitung_gambar(folder):
    """Hitung file gambar langsung di dalam folder (tidak rekursif, sama seperti
    yang nanti dilihat oleh dataset.py)."""
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EKSTENSI_GAMBAR
    )


def cek_dataset(root):
    """Pastikan keempat folder ada dan tidak kosong. Mengembalikan dict jumlah gambar."""
    if not root.is_dir():
        raise DataError(f"Folder dataset tidak ditemukan: {root}")

    hilang = [n for n in SUBFOLDER if not (root / n).is_dir()]
    if hilang:
        raise DataError(
            f"Dataset di {root} tidak lengkap. Folder yang hilang: {', '.join(hilang)}."
        )

    jumlah = {n: hitung_gambar(root / n) for n in SUBFOLDER}
    kosong = [n for n, j in jumlah.items() if j == 0]
    if kosong:
        raise DataError(
            f"Folder ada tetapi tidak berisi gambar (.jpg/.jpeg/.png): {', '.join(kosong)}."
        )
    return jumlah


def cetak_tabel(jumlah):
    print()
    print(f"{'Folder':<8}{'Jumlah gambar':>15}   Keterangan")
    print("-" * 60)
    for nama in SUBFOLDER:
        print(f"{nama:<8}{jumlah[nama]:>15,}   {KETERANGAN[nama]}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Unduhan
# ---------------------------------------------------------------------------
def _cetak_progres(terunduh, total):
    mb = terunduh / 1024 / 1024
    if total:
        persen = terunduh * 100 / total
        sys.stdout.write(f"\r  {mb:7.1f} MB / {total / 1024 / 1024:.1f} MB ({persen:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {mb:7.1f} MB")
    sys.stdout.flush()


def _unduh_satu(url, tujuan):
    """Satu kali percobaan. Data ditulis ke file .part dan baru di-rename setelah
    ukurannya cocok, supaya unduhan yang terputus tidak tampak seperti zip valid."""
    sementara = tujuan.with_name(tujuan.name + ".part")
    permintaan = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (download_data.py)"})
    try:
        with urllib.request.urlopen(permintaan, timeout=TIMEOUT_DETIK) as resp, \
                open(sementara, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            terunduh = 0
            while True:
                potongan = resp.read(UKURAN_CHUNK)
                if not potongan:
                    break
                f.write(potongan)
                terunduh += len(potongan)
                _cetak_progres(terunduh, total)
    except BaseException:
        print()  # pindah baris dari baris progres
        sementara.unlink(missing_ok=True)
        raise
    print()

    # Content-Length tersedia -> pastikan semua byte benar-benar diterima
    if total and terunduh != total:
        sementara.unlink(missing_ok=True)
        raise ConnectionError(f"unduhan terputus ({terunduh} dari {total} byte)")
    sementara.replace(tujuan)


def unduh_dengan_cadangan(daftar_url, tujuan):
    """Coba tiap alamat sampai berhasil; ulangi beberapa kali untuk gangguan sementara."""
    catatan = []
    for url in daftar_url:
        for percobaan in range(1, JUMLAH_PERCOBAAN + 1):
            print(f"Mengunduh (percobaan {percobaan}/{JUMLAH_PERCOBAAN}): {url}")
            try:
                _unduh_satu(url, tujuan)
                return
            except urllib.error.HTTPError as e:
                alasan = f"server membalas HTTP {e.code} ({e.reason})"
                # 403/404/410 berarti alamatnya memang bermasalah, mengulang tidak membantu
                ulangi = e.code not in (403, 404, 410)
            except urllib.error.URLError as e:
                alasan = f"tidak dapat terhubung ({e.reason})"
                ulangi = True
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    raise DataError(
                        "Disk penuh saat menyimpan unduhan. Kosongkan ruang disk, "
                        "lalu jalankan ulang skrip ini."
                    ) from None
                alasan = f"koneksi terputus atau file tidak bisa ditulis ({e})"
                ulangi = True

            print(f"  Gagal: {alasan}")
            catatan.append(f"{url}\n    -> {alasan}")
            if not ulangi:
                break
            if percobaan < JUMLAH_PERCOBAAN:
                time.sleep(2 * percobaan)  # jeda bertahap: 2 detik, 4 detik, ...

    raise DataError(
        "Semua alamat unduhan gagal:\n  " + "\n  ".join(catatan) + "\n\n"
        "Yang bisa dilakukan:\n"
        "  1. Periksa koneksi internet (atau proxy kampus), lalu jalankan ulang skrip ini.\n"
        "  2. Unduh manual lewat browser dari salah satu alamat di atas, lalu jalankan:\n"
        f"     python download_data.py --zip-path <lokasi file {NAMA_DATASET}.zip>"
    )


# ---------------------------------------------------------------------------
# Ekstraksi
# ---------------------------------------------------------------------------
def ekstrak_aman(zip_path, staging):
    """Uji integritas zip, tolak path yang mencoba keluar dari folder tujuan
    (zip-slip), lalu ekstrak ke folder staging."""
    if not zipfile.is_zipfile(zip_path):
        raise DataError(
            f"File ini bukan zip yang valid atau rusak: {zip_path}\n"
            "Hapus file tersebut lalu jalankan ulang, atau unduh ulang dengan --force."
        )

    staging = staging.resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            print("Menguji integritas zip ...")
            rusak = zf.testzip()
            if rusak is not None:
                raise DataError(
                    f"Zip rusak (berkas bermasalah: {rusak}). "
                    "Unduh ulang dengan --force."
                )

            for anggota in zf.infolist():
                target = (staging / anggota.filename).resolve()
                try:
                    target.relative_to(staging)
                except ValueError:
                    raise DataError(
                        f"Zip berisi path yang tidak aman: {anggota.filename}. "
                        "Ekstraksi dibatalkan."
                    ) from None

            print("Mengekstrak ...")
            zf.extractall(staging)
    except zipfile.BadZipFile as e:
        raise DataError(f"Zip tidak bisa dibaca ({e}). Unduh ulang dengan --force.") from None


def cari_root_dataset(staging):
    """Cari folder yang memuat keempat subfolder. Dengan begini zip boleh punya
    folder induk (monet2photo/) atau tidak, dan sampah macOS (__MACOSX) diabaikan."""
    for kandidat in staging.rglob("trainA"):
        if "__MACOSX" in kandidat.parts or not kandidat.is_dir():
            continue
        induk = kandidat.parent
        if all((induk / nama).is_dir() for nama in SUBFOLDER):
            return induk
    return None


def siapkan_dari_zip(zip_path, root):
    """Ekstrak ke folder staging dulu, baru dipindah ke root. Jika ada langkah yang
    gagal, root tidak pernah berisi hasil ekstraksi setengah jadi."""
    staging = root.parent / "_staging_ekstrak"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        ekstrak_aman(zip_path, staging)
        asal = cari_root_dataset(staging)
        if asal is None:
            raise DataError(
                f"Zip berhasil diekstrak tetapi tidak berisi keempat folder "
                f"({', '.join(SUBFOLDER)}). Pastikan file zip yang dipakai adalah "
                f"{NAMA_DATASET}.zip yang benar."
            )
        if root.exists():
            shutil.rmtree(root)  # hanya sampai di sini jika pengguna memakai --force
        shutil.move(str(asal), str(root))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Alur utama
# ---------------------------------------------------------------------------
def jalankan(args, data_dir, root):
    # Mode cek saja: tidak mengunduh apa pun
    if args.verify_only:
        try:
            jumlah = cek_dataset(root)
        except DataError as e:
            raise DataError(
                f"{e}\nJalankan 'python download_data.py' tanpa --verify-only untuk menyiapkannya."
            ) from None
        cetak_tabel(jumlah)
        print("Dataset lengkap.")
        return

    # Idempoten: dataset yang sudah lengkap tidak diunduh ulang.
    # Folder yang tidak lengkap tidak dihapus tanpa izin eksplisit (--force).
    if root.exists() and not args.force:
        try:
            jumlah = cek_dataset(root)
        except DataError as e:
            raise DataError(
                f"{e}\n\nSkrip tidak menghapus folder itu secara otomatis. "
                "Jalankan ulang dengan --force untuk menghapusnya dan menggantinya "
                "dengan hasil unduhan baru."
            ) from None
        cetak_tabel(jumlah)
        print("Dataset sudah lengkap, unduhan dilewati (gunakan --force untuk mengunduh ulang).")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # Tentukan sumber zip: file lokal dari pengguna, atau hasil unduhan
    zip_milik_skrip = False
    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.is_file():
            raise DataError(f"File zip tidak ditemukan: {zip_path}")
    else:
        zip_path = data_dir / f"{NAMA_DATASET}.zip"
        zip_milik_skrip = True
        if zip_path.is_file() and zipfile.is_zipfile(zip_path) and not args.force:
            print(f"Memakai zip yang sudah ada: {zip_path}")
        else:
            daftar_url = (args.url,) if args.url else URL_BAWAAN
            unduh_dengan_cadangan(daftar_url, zip_path)

    siapkan_dari_zip(zip_path, root)

    jumlah = cek_dataset(root)
    cetak_tabel(jumlah)
    print(f"Dataset siap di: {root.resolve()}")

    # Zip hasil unduhan dihapus agar tidak memakan ruang; zip milik pengguna tidak disentuh
    if zip_milik_skrip and not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("File zip dihapus untuk menghemat ruang (gunakan --keep-zip untuk menyimpannya).")


def baca_argumen():
    p = argparse.ArgumentParser(description="Unduh dan siapkan dataset monet2photo untuk CycleGAN.")
    p.add_argument("--data-dir", default=None,
                   help="folder tujuan (default: folder 'data' di samping skrip ini)")
    p.add_argument("--url", default=None,
                   help="alamat zip khusus; jika diisi, alamat bawaan tidak dipakai")
    p.add_argument("--zip-path", default=None,
                   help="pakai file zip lokal (hasil unduhan manual) tanpa mengunduh")
    p.add_argument("--force", action="store_true",
                   help="unduh ulang dan ganti folder dataset yang sudah ada")
    p.add_argument("--verify-only", action="store_true",
                   help="hanya verifikasi folder dataset yang sudah ada")
    p.add_argument("--keep-zip", action="store_true",
                   help="jangan hapus file zip setelah ekstraksi")
    return p.parse_args()


def main():
    args = baca_argumen()
    data_dir = Path(args.data_dir) if args.data_dir else DIR_BAWAAN
    root = data_dir / NAMA_DATASET

    try:
        jalankan(args, data_dir, root)
    except DataError as e:
        print(f"\n[GAGAL] {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            pesan = "Disk penuh. Kosongkan ruang disk lalu jalankan ulang skrip ini."
        else:
            pesan = f"Terjadi kesalahan sistem file: {e}"
        print(f"\n[GAGAL] {pesan}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDibatalkan oleh pengguna.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()