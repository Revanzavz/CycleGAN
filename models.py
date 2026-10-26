"""
models.py
Arsitektur CycleGAN: Generator (ResNet) dan Discriminator (PatchGAN 70x70).

Konvensi proyek:
- Input dan output gambar berukuran (3, 128, 128) dengan rentang [-1, 1]
- G_AB: Monet -> Foto, G_BA: Foto -> Monet
- D_A menilai domain Monet, D_B menilai domain Foto
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Inisialisasi bobot
# ---------------------------------------------------------------------------
def init_weights(net, mean=0.0, std=0.02):
    """Inisialisasi bobot konvolusi ~ N(mean, std). Dipanggil lewat net.apply()."""

    def _init(m):
        # Hanya lapisan konvolusi yang punya bobot yang perlu diinisialisasi,
        # InstanceNorm2d kita pakai tanpa parameter (affine=False)
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean, std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.apply(_init)
    return net


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Dua konvolusi 3x3 dengan koneksi skip: keluaran = x + F(x)."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # padding refleksi agar tepi gambar tidak berpinggiran hitam
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            # tanpa ReLU di akhir: residu boleh bernilai negatif
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection menjaga struktur gambar asli


class Generator(nn.Module):
    """
    Generator ResNet:
      7x7 conv -> 2x downsampling -> N residual block -> 2x upsampling -> 7x7 conv + tanh
    Bentuk: (N, 3, 128, 128) -> (N, 3, 128, 128)
    """

    def __init__(self, in_channels=3, out_channels=3, base_filters=64, n_res_blocks=6):
        super().__init__()

        # Blok awal: refleksi padding + konvolusi 7x7 (64 filter)
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]

        # Dua blok downsampling: resolusi /2, jumlah filter x2 (64 -> 128 -> 256)
        ch = base_filters
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2

        # Residual block bekerja di resolusi 32x32 dengan 256 kanal
        layers += [ResidualBlock(ch) for _ in range(n_res_blocks)]

        # Dua blok upsampling: Upsample lalu Conv (bukan ConvTranspose)
        # agar tidak muncul artefak checkerboard. Filter 256 -> 128 -> 64
        for _ in range(2):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch // 2, kernel_size=3, bias=False),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2

        # Blok akhir: konvolusi 7x7 lalu tanh agar keluaran berada di [-1, 1]
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ch, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        init_weights(self)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Diskriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    PatchGAN 70x70 (kernel 4x4). Setiap nilai pada peta keluaran menilai
    satu patch 70x70 piksel dari gambar masukan.
    Bentuk: (N, 3, 128, 128) -> (N, 1, 14, 14), tanpa sigmoid (untuk LSGAN).
    """

    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()
        f = base_filters

        self.model = nn.Sequential(
            # Layer 1: tanpa normalisasi (64x64)
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 -> 128 filter (32x32)
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 128 -> 256 filter (16x16)
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: stride 1, 256 -> 512 filter (15x15)
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer akhir: peta skor satu kanal (14x14), tanpa sigmoid
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
        )
        init_weights(self)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Pengujian cepat: python models.py
# ---------------------------------------------------------------------------
def _count_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 128, 128, device=device)  # batch acak (2, 3, 128, 128)

    G = Generator().to(device)
    D = Discriminator().to(device)

    with torch.no_grad():
        y = G(x)
        s = D(y)

    print(f"Device               : {device}")
    print(f"Generator  input     : {tuple(x.shape)}")
    print(f"Generator  output    : {tuple(y.shape)}  (range [{y.min():.3f}, {y.max():.3f}])")
    print(f"Generator  parameter : {_count_params(G):,}")
    print(f"Diskriminator input  : {tuple(y.shape)}")
    print(f"Diskriminator output : {tuple(s.shape)}")
    print(f"Diskriminator param. : {_count_params(D):,}")

    assert y.shape == x.shape, "Output generator harus sama dengan input"
    assert y.min() >= -1.0 and y.max() <= 1.0, "Output generator harus di [-1, 1]"
    assert s.shape == (2, 1, 14, 14), "Peta skor diskriminator harus (N, 1, 14, 14)"
    print("Semua pengecekan lolos.")
