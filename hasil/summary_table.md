Nilai loss = rata-rata 5 epoch terakhir. rec_err = cycle reconstruction error (L1) pada data test.

| experiment | epochs | loss_G | loss_D | loss_cycle | loss_identity | loss_G_adv | D_real_score | D_fake_score | menit_per_epoch | rec_err_A | rec_err_B | rec_err_mean | catatan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cyc10_id5_bs1 | 10 | 7.8717 | 0.3397 | 0.4673 | 0.4511 | 0.9438 | 0.71 | 0.2936 | 1.01 | 0.2005 | 0.217 | 0.2088 | normal |
| cyc10_id0_bs1 | 10 | 6.1386 | 0.3679 | 0.5279 | 0.5284 | 0.8593 | 0.6829 | 0.3174 | 0.68 | 0.2092 | 0.2301 | 0.2197 | normal |
| cyc10_id5_bs2 | 10 | 7.3514 | 0.3323 | 0.4325 | 0.4191 | 0.9312 | 0.7049 | 0.2963 | 1.23 | 0.1913 | 0.1902 | 0.1908 | normal |
| cyc15_id5_bs1 | 10 | 10.2337 | 0.3101 | 0.4631 | 0.4487 | 1.0429 | 0.7402 | 0.2633 | 0.98 | 0.1946 | 0.2096 | 0.2021 | normal |
| cyc5_id5_bs1 | 10 | 5.3581 | 0.3942 | 0.4695 | 0.4384 | 0.8184 | 0.6555 | 0.3477 | 0.74 | 0.2031 | 0.2107 | 0.2069 | normal |
