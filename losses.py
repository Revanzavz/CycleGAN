"""
losses.py
Fungsi-fungsi loss untuk CycleGAN (Monet <-> Foto).

Konvensi proyek:
- Domain A = lukisan Monet, domain B = foto.
- G_AB: Monet -> Foto, G_BA: Foto -> Monet.
- D_A menilai "apakah ini lukisan Monet asli", D_B menilai "apakah ini foto asli".
- Diskriminator mengeluarkan peta nilai 1 kanal TANPA sigmoid (LSGAN).

Isi file:
1. Adversarial loss berbasis MSE (LSGAN).
2. Cycle-consistency loss (L1), dikali lambda_cycle.
3. Identity loss (L1), dikali lambda_identity.

Catatan bobot: di paper asli identity loss = 0.5 * lambda_cycle. Di proyek ini
kita langsung pakai lambda_identity = 5 (= 0.5 * 10) supaya lebih mudah diatur.
"""

import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """Adversarial loss gaya LSGAN: MSE antara peta prediksi D dan target 1 (asli) / 0 (palsu).

    Kenapa MSE dan bukan BCE? Dengan BCE + sigmoid, gradien hampir nol ketika D sudah
    yakin. MSE tetap memberi gradien yang sebanding dengan jarak ke target, sehingga
    training GAN lebih stabil.
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        # Hitung dalam float32 supaya aman saat mixed precision (output D bisa float16)
        prediction = prediction.float()
        value = self.real_label if target_is_real else self.fake_label
        # Target dibuat seukuran peta patch dari PatchGAN
        target = torch.full_like(prediction, value)
        return self.mse(prediction, target)


class CycleGANLoss(nn.Module):
    """Kumpulan semua loss CycleGAN dengan bobot yang bisa diatur dari luar.

    Contoh:
        crit = CycleGANLoss(lambda_cycle=10, lambda_identity=5)
        loss_adv = crit.adversarial(D_B(fake_B), True)
        loss_cyc, raw_cyc = crit.cycle(real_A, rec_A)
    """

    def __init__(self, lambda_cycle: float = 10.0, lambda_identity: float = 5.0):
        super().__init__()
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_identity = float(lambda_identity)
        self.adversarial = AdversarialLoss()
        self.l1 = nn.L1Loss()

    def cycle(self, real: torch.Tensor, reconstructed: torch.Tensor):
        """Cycle-consistency: real -> G1 -> G2 -> harus kembali mirip real.

        Mengembalikan (loss_berbobot, loss_mentah). Loss mentah (tanpa lambda)
        dipakai untuk logging supaya eksperimen dengan lambda berbeda tetap bisa
        dibandingkan secara adil.
        """
        raw = self.l1(reconstructed.float(), real.float())
        return self.lambda_cycle * raw, raw

    def identity(self, real: torch.Tensor, same: torch.Tensor):
        """Identity: G_AB(real_B) sebaiknya tetap real_B (gambar yang sudah berada di
        domain target tidak perlu diubah). Membantu menjaga warna dan komposisi.

        Mengembalikan (loss_berbobot, loss_mentah).
        """
        raw = self.l1(same.float(), real.float())
        return self.lambda_identity * raw, raw

    def discriminator(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        """Loss diskriminator: real -> 1, fake -> 0. Dikali 0.5 supaya D belajar
        lebih lambat dibanding G (sesuai paper CycleGAN)."""
        return 0.5 * (self.adversarial(pred_real, True) + self.adversarial(pred_fake, False))


if __name__ == "__main__":
    # Uji cepat dengan tensor acak
    crit = CycleGANLoss(lambda_cycle=10, lambda_identity=5)
    pred = torch.randn(2, 1, 14, 14)
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    y = torch.rand(2, 3, 128, 128) * 2 - 1
    print("adv (target real) :", crit.adversarial(pred, True).item())
    print("adv (target fake) :", crit.adversarial(pred, False).item())
    w, r = crit.cycle(x, y)
    print(f"cycle berbobot={w.item():.4f}, mentah={r.item():.4f}")
    w, r = crit.identity(x, x)
    print(f"identity (gambar sama) berbobot={w.item():.4f} (harus 0)")
    print("loss D:", crit.discriminator(pred, pred).item())
