import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# Basic residual block
##############################################################################
class BasicResBlock(nn.Module):
    """
    A 2-layer residual block with BN and ReLU.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(residual + out)


##############################################################################
# Vector quantization
##############################################################################
class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with EMA updates (for VQ-VAE).
    """
    def __init__(self, n_embed=1024, embed_dim=256, beta=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        # embeddings, cluster usage, average
        self.register_buffer('embedding', torch.randn(n_embed, embed_dim))
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', self.embedding.clone())

    def forward(self, z_e):
        """
        z_e: (B, embed_dim, H, W)
        returns z_q (quantized), vq_loss
        """
        B, C, H, W = z_e.shape
        flat_z = z_e.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # compute distance to codebook
        dist = (flat_z ** 2).sum(dim=1, keepdim=True) \
             - 2 * flat_z @ self.embedding.T \
             + (self.embedding.T ** 2).sum(dim=0, keepdim=True)
        min_indices = dist.argmin(dim=1)  # (B*H*W,)

        z_q = self.embedding[min_indices].view(B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2)  # => (B, C, H, W)

        # EMA update
        if self.training:
            one_hot = F.one_hot(min_indices, self.n_embed).float()
            cluster_size = one_hot.sum(dim=0)
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=(1-self.decay))

            embed_sum = one_hot.T @ flat_z
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=(1-self.decay))

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.n_embed*self.epsilon) * n
            self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        # VQ loss
        vq_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
        # straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, vq_loss


##############################################################################
# Bigger, deeper VQ-VAE for 256x256 -> 8x8 latent
##############################################################################
class BiggerDeeperVQVAE(nn.Module):
    def __init__(self, in_ch=1, base_ch=128, embed_dim=256, n_embed=1024):
        super().__init__()
        # ------ Encoder ------
        enc_layers = []
        # 256->128
        enc_layers += [
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        ]
        # 128->64
        enc_layers += [
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        ]
        # 64->32
        enc_layers += [
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        ]
        # 32->16
        enc_layers += [
            nn.Conv2d(base_ch*4, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        ]
        # 16->8
        enc_layers += [
            nn.Conv2d(base_ch*4, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        ]

        # more residual blocks
        for _ in range(4):
            enc_layers.append(BasicResBlock(base_ch*4))

        self.need_1x1 = (embed_dim != base_ch*4)
        if self.need_1x1:
            enc_layers += [
                nn.Conv2d(base_ch*4, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ]
        self.encoder = nn.Sequential(*enc_layers)

        # vector quantizer
        self.vq = VectorQuantizer(n_embed=n_embed, embed_dim=embed_dim)

        # ------ Decoder ------
        dec_layers = []
        if self.need_1x1:
            dec_layers += [
                nn.Conv2d(embed_dim, base_ch*4, 1),
                nn.BatchNorm2d(base_ch*4),
                nn.ReLU(inplace=True)
            ]

        for _ in range(4):
            dec_layers.append(BasicResBlock(base_ch*4))

        # 8->16
        dec_layers += [
            nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        ]
        # 16->32
        dec_layers += [
            nn.ConvTranspose2d(base_ch*4, base_ch*4, 4, 2, 1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        ]
        # 32->64
        dec_layers += [
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        ]
        # 64->128
        dec_layers += [
            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        ]
        # 128->256
        dec_layers += [
            nn.ConvTranspose2d(base_ch, in_ch, 4, 2, 1)
        ]

        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq(z_e)
        return z_e, z_q, vq_loss

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        """
        x -> encode -> quantize -> decode
        """
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss
