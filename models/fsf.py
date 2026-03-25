import torch
import torch.nn as nn
import torch.nn.functional as F


class FSFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, low_freq_ratio=0.25):
        super().__init__()

        self.low_freq_ratio = low_freq_ratio

        # Learnable frequency filter ω
        self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        # Channel reduction after fusion
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x, skip):
        """
        x: decoder feature
        skip: encoder feature
        """

        # ----- 1. Match spatial size -----
        if x.size()[2:] != skip.size()[2:]:
            skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)

        # ----- 2. Combine (temporary fusion) -----
        X = torch.cat([x, skip], dim=1)   # (B, C, H, W)

        # ----- 3. FFT -----
        Freq = torch.fft.fft2(X, norm='ortho')

        # ----- 4. Apply learnable filter ω -----
        Freq = Freq * self.weight

        # ----- 5. Low-frequency region -----
        B, C, H, W = Freq.shape
        h_low = int(H * self.low_freq_ratio)
        w_low = int(W * self.low_freq_ratio)

        F_low = Freq[:, :, :h_low, :w_low]

        # ----- 6. Compute mean & std -----
        mu = F_low.mean(dim=(2, 3), keepdim=True)
        sigma = F_low.std(dim=(2, 3), keepdim=True) + 1e-6

        # ----- 7. Resampling (training only) -----
        if self.training:
            eps_mu = torch.randn_like(mu)
            eps_sigma = torch.randn_like(sigma)

            mu_hat = mu + eps_mu * sigma
            sigma_hat = sigma + eps_sigma * sigma

            F_low = (F_low - mu) / sigma
            F_low = F_low * sigma_hat + mu_hat

        # ----- 8. Replace low-frequency -----
        # Create a copy of Freq and replace low-frequency region
        Freq_new = Freq.clone()
        Freq_new[:, :, :h_low, :w_low] = F_low
        Freq = Freq_new

        # ----- 9. IFFT -----
        out = torch.fft.ifft2(Freq, norm='ortho').real

        # ----- 10. Channel reduction (VERY IMPORTANT) -----
        out = self.channel_reduce(out)

        return out