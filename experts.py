import torch
import torch.nn as nn
import torch.nn.functional as F


##############################################################################
# TimeEmbedding
##############################################################################
class TimeEmbedding(nn.Module):
    """
    Embedding for integer timesteps t in [0, T-1].
    Then an MLP to produce (B, emb_dim).
    """
    def __init__(self, dim, timesteps):
        super().__init__()
        self.embed = nn.Embedding(timesteps, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        """
        t: shape=(B,) with integer steps
        returns => (B,dim)
        """
        x = self.embed(t)
        return self.mlp(x)


##############################################################################
# Some building blocks: NonLocalBlock, ChannelAttention, etc.
##############################################################################
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.inter_ch = in_channels // 2
        self.g = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_ch, 1)
        self.W = nn.Conv2d(self.inter_ch, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        g_x = self.g(x).view(B, self.inter_ch, -1)     # (B, inter_ch, H*W)
        theta_x = self.theta(x).view(B, self.inter_ch, -1)
        phi_x = self.phi(x).view(B, self.inter_ch, -1)

        f = torch.matmul(theta_x.transpose(1,2), phi_x)  # (B,H*W,H*W)
        f_div = F.softmax(f, dim=-1)
        y = torch.matmul(f_div, g_x.transpose(1,2))      # (B,H*W,inter_ch)
        y = y.transpose(1,2).view(B, self.inter_ch, H, W)
        return x + self.W(y)


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        scale = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class EdgeEnhancementBlock(nn.Module):
    """
    A block that enhances edges using a Laplacian kernel
    """
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor(
            [[0,-1,0],
             [-1,4,-1],
             [0,-1,0]],
            dtype=torch.float32
        ).view(1,1,3,3)
        self.register_buffer("lap_kernel", kernel)
        self.merge = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            ec = F.conv2d(x[:,c:c+1], self.lap_kernel, padding=1)
            edges.append(ec)
        edges = torch.cat(edges, dim=1)
        out = x + edges
        return self.merge(out)


class HighFreqAttentionBlock(nn.Module):
    """
    Depthwise high-pass + conv => attention map => multiply with input
    """
    def __init__(self, channels):
        super().__init__()
        self.hp_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        with torch.no_grad():
            kernel = torch.tensor(
                [[-1,-1,-1],
                 [-1, 8,-1],
                 [-1,-1,-1]], dtype=torch.float32
            ).view(1,1,3,3).expand(channels,1,3,3)
            self.hp_conv.weight.copy_(kernel)
        self.hp_conv.weight.requires_grad = False

        self.att_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hp = self.hp_conv(x)
        att_map = self.sigmoid(self.att_conv(hp))
        return x * att_map


##############################################################################
# FiLM ResBlock (with time embedding)
##############################################################################
class FiLMResBlock(nn.Module):
    """
    A block that can incorporate time embedding (FiLM style).
    """
    def __init__(self, channels, emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)

        if emb_dim is not None:
            self.emb_proj = nn.Linear(emb_dim, channels)
        else:
            self.emb_proj = None

    def forward(self, x, emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if self.emb_proj is not None and emb is not None:
            gamma = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)  # (B, channels,1,1)
            h = h + gamma

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.ca(h)

        return x + h


def forward_film_sequential(layers, x, emb):
    """
    Utility: pass (x, emb) through a sequence of layers
    if layer is FiLMResBlock, pass emb, else pass only x.
    """
    for layer in layers:
        if isinstance(layer, FiLMResBlock):
            x = layer(x, emb)
        else:
            x = layer(x)
    return x


##############################################################################
# Expert networks
##############################################################################
class ExpertNetE1(nn.Module):
    """
    Expert1: Smooth white matter => includes an EdgeEnhancementBlock in the bottleneck
    """
    def __init__(self, base_channels=256, timesteps=100):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels, timesteps)
        self.fuse_in = nn.Conv2d(base_channels*2 + base_channels, base_channels, 3, padding=1)

        # encoder
        self.enc1 = [FiLMResBlock(base_channels, emb_dim=base_channels),
                     FiLMResBlock(base_channels, emb_dim=base_channels)]
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)

        self.enc2 = [FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     FiLMResBlock(base_channels*2, emb_dim=base_channels)]
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)

        self.enc3 = [FiLMResBlock(base_channels*4, emb_dim=base_channels)]

        # bottleneck (edge enhancement)
        self.mid = [
            FiLMResBlock(base_channels*4, emb_dim=base_channels),
            EdgeEnhancementBlock(base_channels*4),
            FiLMResBlock(base_channels*4, emb_dim=base_channels)
        ]

        # decoder
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.dec1 = [FiLMResBlock(base_channels*4, emb_dim=base_channels),
                     FiLMResBlock(base_channels*4, emb_dim=base_channels),
                     nn.Conv2d(base_channels*4, base_channels*2, 1)]

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1)
        self.dec2 = [FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     nn.Conv2d(base_channels*2, base_channels, 1)]

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 1)
        )

    def forward(self, z_t, cond, t):
        B, C, H, W = z_t.shape
        t_emb = self.time_embed(t)
        t_2d = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        fused = torch.cat([z_t, cond, t_2d], dim=1)
        fused = self.fuse_in(fused)

        e1 = forward_film_sequential(self.enc1, fused, t_emb)
        e2_in = self.down1(e1)
        e2 = forward_film_sequential(self.enc2, e2_in, t_emb)
        e3_in = self.down2(e2)
        e3 = forward_film_sequential(self.enc3, e3_in, t_emb)

        m = forward_film_sequential(self.mid, e3, t_emb)

        d1_in = self.up1(m)
        d1_cat = torch.cat([d1_in, e2], dim=1)
        d1 = forward_film_sequential(self.dec1, d1_cat, t_emb)

        d2_in = self.up2(d1)
        d2_cat = torch.cat([d2_in, e1], dim=1)
        d2 = forward_film_sequential(self.dec2, d2_cat, t_emb)

        return self.out(d2)


class ExpertNetE2(nn.Module):
    """
    Expert2: focus on cortical edges => includes a HighFreqAttentionBlock
    """
    def __init__(self, base_channels=256, timesteps=100):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels, timesteps)
        self.fuse_in = nn.Conv2d(base_channels*2 + base_channels, base_channels, 3, padding=1)

        # encoder
        self.enc1 = [FiLMResBlock(base_channels, emb_dim=base_channels),
                     FiLMResBlock(base_channels, emb_dim=base_channels)]
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)

        self.enc2 = [FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     FiLMResBlock(base_channels*2, emb_dim=base_channels)]
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)

        # bottleneck
        self.mid = [
            FiLMResBlock(base_channels*4, emb_dim=base_channels),
            HighFreqAttentionBlock(base_channels*4),
            FiLMResBlock(base_channels*4, emb_dim=base_channels)
        ]

        # decoder
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.dec1 = [FiLMResBlock(base_channels*4, emb_dim=base_channels),
                     FiLMResBlock(base_channels*4, emb_dim=base_channels),
                     nn.Conv2d(base_channels*4, base_channels*2, 1)]

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1)
        self.dec2 = [FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     nn.Conv2d(base_channels*2, base_channels, 1)]

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 1)
        )

    def forward(self, z_t, cond, t):
        B, C, H, W = z_t.shape
        t_emb = self.time_embed(t)
        t_2d = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        fused = torch.cat([z_t, cond, t_2d], dim=1)
        fused = self.fuse_in(fused)

        e1 = forward_film_sequential(self.enc1, fused, t_emb)
        e2_in = self.down1(e1)
        e2 = forward_film_sequential(self.enc2, e2_in, t_emb)
        b_in = self.down2(e2)

        b = forward_film_sequential(self.mid, b_in, t_emb)

        d1_in = self.up1(b)
        d1_cat = torch.cat([d1_in, e2], dim=1)
        d1 = forward_film_sequential(self.dec1, d1_cat, t_emb)

        d2_in = self.up2(d1)
        d2_cat = torch.cat([d2_in, e1], dim=1)
        d2 = forward_film_sequential(self.dec2, d2_cat, t_emb)
        return self.out(d2)


class ExpertNetE3(nn.Module):
    """
    Expert3: focus on grey-white matter junction => non-local attention in the decoder
    """
    def __init__(self, base_channels=256, timesteps=100):
        super().__init__()
        self.time_embed = TimeEmbedding(base_channels, timesteps)
        self.fuse_in = nn.Conv2d(base_channels*2 + base_channels, base_channels, 3, padding=1)

        # encoder
        self.enc1 = [FiLMResBlock(base_channels, emb_dim=base_channels),
                     FiLMResBlock(base_channels, emb_dim=base_channels)]
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 4, 2, 1)

        self.enc2 = [FiLMResBlock(base_channels*2, emb_dim=base_channels),
                     FiLMResBlock(base_channels*2, emb_dim=base_channels)]
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1)

        # bottleneck
        self.mid = [
            FiLMResBlock(base_channels*4, emb_dim=base_channels),
            FiLMResBlock(base_channels*4, emb_dim=base_channels)
        ]

        # decoder
        self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)
        self.dec1 = [
            FiLMResBlock(base_channels*4, emb_dim=base_channels),
            FiLMResBlock(base_channels*4, emb_dim=base_channels),
            NonLocalBlock(base_channels*4),
            nn.Conv2d(base_channels*4, base_channels*2, 1)
        ]

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1)
        self.dec2 = [
            FiLMResBlock(base_channels*2, emb_dim=base_channels),
            FiLMResBlock(base_channels*2, emb_dim=base_channels),
            nn.Conv2d(base_channels*2, base_channels, 1)
        ]

        self.out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 1)
        )

    def forward(self, z_t, cond, t):
        B, C, H, W = z_t.shape
        t_emb = self.time_embed(t)
        t_2d = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        fused = torch.cat([z_t, cond, t_2d], dim=1)
        fused = self.fuse_in(fused)

        e1 = forward_film_sequential(self.enc1, fused, t_emb)
        e2_in = self.down1(e1)
        e2 = forward_film_sequential(self.enc2, e2_in, t_emb)

        b_in = self.down2(e2)
        b = forward_film_sequential(self.mid, b_in, t_emb)

        d1_in = self.up1(b)
        d1_cat = torch.cat([d1_in, e2], dim=1)
        d1 = forward_film_sequential(self.dec1, d1_cat, t_emb)

        d2_in = self.up2(d1)
        d2_cat = torch.cat([d2_in, e1], dim=1)
        d2 = forward_film_sequential(self.dec2, d2_cat, t_emb)

        return self.out(d2)
