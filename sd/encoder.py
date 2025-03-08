import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlcok


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResidualBlcok(128, 128),
            VAE_ResidualBlcok(128, 128),
            # 1/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlcok(128, 256),
            VAE_ResidualBlcok(256, 256),
            # 1/4
            nn.Conv2D(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlcok(256, 512),
            VAE_ResidualBlcok(512, 512),
            # 1/8
            nn.Conv2D(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlcok(512, 512),
            VAE_ResidualBlcok(512, 512),

            VAE_ResidualBlcok(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlcok(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2D(8, 8, kernel_size=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (L, R, T, B)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)

        # (batch, 8, H/8, W/8) -> [(Batch, 4, H/8, W,8), (Batch, 4, H/8, W,8)]
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)

        var = log_var.exp()

        std = var.sqrt()

        # z = N(0,1) -> N(mean, std)?
        # sampling from (mean, std)
        x = mean + std * noise

        # scale constant value(????)
        x *= 0.18215

        return x