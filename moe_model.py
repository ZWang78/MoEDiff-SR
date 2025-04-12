import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from experts import ExpertNetE1, ExpertNetE2, ExpertNetE3
from vqvae import BiggerDeeperVQVAE

# timm for SwinTransformerBlock
from timm.models.swin_transformer import SwinTransformerBlock


##############################################################################
# 1) Multi-scale patch embedding + Swin for 3T encoding
##############################################################################
class MultiScaleSwin3TEncoder(nn.Module):
    """
    This class outputs a sequence of tokens (B, L, embed_dim).
    Example patch sizes: 64x64, 32x32, 16x16 for a 256x256 input.
    """

    def __init__(self,
                 in_ch=1,
                 embed_dim=64,
                 num_heads=4,
                 window_size=4,
                 depth=2):
        super().__init__()
        # multi-scale conv projections
        self.proj_large = nn.Conv2d(in_ch, embed_dim, kernel_size=64, stride=64)
        self.proj_medium = nn.Conv2d(in_ch, embed_dim, kernel_size=32, stride=32)
        self.proj_small = nn.Conv2d(in_ch, embed_dim, kernel_size=16, stride=16)

        # Swin blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim,
                input_resolution=(None, None),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size//2,
                mlp_ratio=4.0
            )
            for i in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        # large
        fl = self.proj_large(x)               # (B,embed_dim,H/64,W/64)
        fl = fl.flatten(2).transpose(1,2)     # (B,N_large,embed_dim)

        # medium
        fm = self.proj_medium(x)              # (B,embed_dim,H/32,W/32)
        fm = fm.flatten(2).transpose(1,2)     # (B,N_medium,embed_dim)

        # small
        fs = self.proj_small(x)               # (B,embed_dim,H/16,W/16)
        fs = fs.flatten(2).transpose(1,2)     # (B,N_small,embed_dim)

        tokens = torch.cat([fl, fm, fs], dim=1)  # (B,N_total,embed_dim)

        # pass tokens into Swin blocks
        # we do a (B,1,L,embed_dim) trick to pass them in
        B, L, D = tokens.shape
        x_swin = tokens.view(B, 1, L, D)
        for blk in self.swin_blocks:
            x_swin = blk(x_swin, input_resolution=(1,L))
        # flatten back
        x_swin = x_swin.view(B, L, D)

        return x_swin  # (B, N_total, embed_dim)


##############################################################################
# 2) MultiScaleSwinGating
##############################################################################
class MultiScaleSwinGating(nn.Module):
    """
    A gating module that uses the same multi-scale patch + Swin approach,
    then outputs gating probabilities for k experts.
    """

    def __init__(self,
                 in_ch=1,
                 embed_dim=64,
                 num_heads=4,
                 window_size=4,
                 depth=2,
                 k_experts=3):
        super().__init__()
        self.feature_extractor = MultiScaleSwin3TEncoder(
            in_ch=in_ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            depth=depth
        )
        self.k_experts = k_experts
        self.fc = nn.Linear(embed_dim, k_experts)

    def forward(self, x3T):
        tokens = self.feature_extractor(x3T)  # (B,L,embed_dim)
        # global average over tokens => (B,embed_dim)
        avg_feat = tokens.mean(dim=1)
        logits = self.fc(avg_feat)            # (B,k_experts)
        gating_probs = F.softmax(logits, dim=-1)
        return gating_probs


##############################################################################
# 3) Helper: cosine schedule, etc.
##############################################################################
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    acp = torch.cos(((x/timesteps)+s)/(1+s)*math.pi/2)**2
    acp = acp/acp[0]
    betas = 1. - (acp[1:] / acp[:-1])
    return torch.clamp(betas, min=1e-6, max=0.999)

def median_absolute_deviation(x):
    x_ = x.view(-1)
    med = x_.median()
    devs = (x_ - med).abs()
    mad = devs.median()
    return mad

def cosine_similarity_4d(a, b, eps=1e-8):
    B = a.size(0)
    a_ = a.view(B, -1)
    b_ = b.view(B, -1)
    dot = (a_*b_).sum(dim=1)
    na = a_.norm(dim=1)+eps
    nb = b_.norm(dim=1)+eps
    return dot/(na*nb)


##############################################################################
# 4) The main MoEModel
##############################################################################
class MoEModel(nn.Module):
    def __init__(self,
                 timesteps=100,
                 base_ch=256,
                 k_experts=3,
                 embed_dim=64,
                 gating_heads=4,
                 gating_win=4,
                 gating_depth=2):
        super().__init__()
        self.timesteps = timesteps
        self.base_ch = base_ch
        self.k_experts = k_experts

        # Condition encoder: multi-scale + Swin => returns tokens
        self.cond_enc = MultiScaleSwin3TEncoder(
            in_ch=1,
            embed_dim=embed_dim,
            num_heads=gating_heads,
            window_size=gating_win,
            depth=gating_depth
        )
        # Gating
        self.gating = MultiScaleSwinGating(
            in_ch=1,
            embed_dim=embed_dim,
            num_heads=gating_heads,
            window_size=gating_win,
            depth=gating_depth,
            k_experts=k_experts
        )

        # Experts
        self.expert_e1 = ExpertNetE1(base_channels=base_ch, timesteps=timesteps)
        self.expert_e2 = ExpertNetE2(base_channels=base_ch, timesteps=timesteps)
        self.expert_e3 = ExpertNetE3(base_channels=base_ch, timesteps=timesteps)
        self.experts = nn.ModuleList([self.expert_e1, self.expert_e2, self.expert_e3])

        # diffusion buffers
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (0,0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', acp)
        self.register_buffer('alphas_cumprod_prev', acp_prev)
        self.register_buffer('sqrt_acp', torch.sqrt(acp))
        self.register_buffer('sqrt_1m_acp', torch.sqrt(1.-acp))

        # posterior variance
        posterior_var = betas*(1.-acp_prev)/(1.-acp)
        self.register_buffer('posterior_variance', posterior_var)

        # gating loss variance tracking
        self.register_buffer('gating_loss_ema', torch.tensor(0.0))
        self.register_buffer('gating_loss_var_ema', torch.tensor(1.0))
        self.ema_decay = 0.99

        # For Expert1's perceptual example => partial VGG
        self.vgg = torchvision.models.vgg16_bn(pretrained=True).features[:15].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward_diffusion(self, z0, t):
        noise = torch.randn_like(z0)
        alpha_sqrt = self.sqrt_acp[t].view(-1,1,1,1)
        one_minus = self.sqrt_1m_acp[t].view(-1,1,1,1)
        z_t = alpha_sqrt*z0 + one_minus*noise
        return z_t, noise

    def forward(self, x_3T, z0_clean, t):

        B, _, H, W = x_3T.shape

        # The gating for the experts
        gating_probs = self.gating(x_3T)  # (B,k_experts)

        # Meanwhile, let's produce a "cond_feat" with a simple conv approach or reuse base_ch:
        # For a real approach: either "reshape the tokens" or do a separate aggregator.
        # For brevity, let's do a direct nn.Conv-based aggregator:
        cond_feat = nn.functional.interpolate(x_3T, size=(8,8), mode='area')
        cond_feat = cond_feat.expand(-1, self.base_ch, -1, -1)  # shape (B, base_ch, 8,8)

        # 2) forward diffusion
        z_t, noise_gt = self.forward_diffusion(z0_clean, t)

        # 3) each expert => predicted noise
        eps_preds = []
        for i, expert in enumerate(self.experts):
            e_i = expert(z_t, cond_feat, t)  # (B, base_ch,8,8)
            eps_preds.append(e_i)

        # 4) gating ensemble
        stack_eps = torch.stack(eps_preds, dim=1)  # (B,k,base_ch,8,8)
        gating_5d = gating_probs.view(B, self.k_experts,1,1,1)
        eps_ensemble = (stack_eps*gating_5d).sum(dim=1)

        return {
            'cond_feat': cond_feat,
            'gating_probs': gating_probs,
            'z_t': z_t,
            'noise_gt': noise_gt,
            'eps_preds': eps_preds,
            'eps_ensemble': eps_ensemble,
            't': t
        }

    def _expert_task_loss(self, i, x_pred, x_clean):
        # E1 => MSE + partial VGG
        if i == 0:
            l2 = F.mse_loss(x_pred, x_clean)
            # small VGG
            def norm_vgg(xx):
                xx_01 = (xx+1)/2
                mean = xx_01.new_tensor([0.485,0.456,0.406]).view(1,3,1,1)
                std  = xx_01.new_tensor([0.229,0.224,0.225]).view(1,3,1,1)
                return (xx_01 - mean)/std
            xp3 = x_pred.repeat(1,3,1,1)
            xc3 = x_clean.repeat(1,3,1,1)
            with torch.no_grad():
                feat_c = self.vgg(norm_vgg(xc3))
            feat_p = self.vgg(norm_vgg(xp3))
            perc = F.mse_loss(feat_p, feat_c)
            return l2 + 0.1*perc

        elif i == 1:
            # E2 => gradient difference
            def grad_map(x):
                dx = x[:,:,:,1:] - x[:,:,:,:-1]
                dy = x[:,:,1:,:] - x[:,:,:-1,:]
                return dx.abs().mean() + dy.abs().mean()
            return (grad_map(x_pred) - grad_map(x_clean)).abs()

        else:
            # E3 => freq domain
            fft_p = torch.fft.rfftn(x_pred, dim=(2,3))
            fft_c = torch.fft.rfftn(x_clean, dim=(2,3))
            return F.mse_loss(torch.abs(fft_p), torch.abs(fft_c))

    def compute_loss(self, out_dict, z0_clean, vqvae):
        gating_probs = out_dict['gating_probs']
        z_t = out_dict['z_t']
        noise_gt = out_dict['noise_gt']
        eps_preds = out_dict['eps_preds']
        t = out_dict['t']

        B = z_t.size(0)
        # Weighted sum of expert-specific losses
        # For each expert: diffusion MSE + specialized "task" loss
        alpha = self.alphas_cumprod[t].view(B,1,1,1)
        sqrt_one_minus = torch.sqrt(1.-alpha)
        sqrt_alpha = torch.sqrt(alpha)

        # decode the clean z0 => x_clean
        with torch.no_grad():
            x_clean = vqvae.decode(z0_clean)  # (B,1,256,256)

        expert_losses = []
        z0_est_list = []
        for i in range(len(eps_preds)):
            diff_loss = F.mse_loss(eps_preds[i], noise_gt)
            z0_est_i = (z_t - sqrt_one_minus*eps_preds[i])/(sqrt_alpha+1e-8)
            z0_est_list.append(z0_est_i)

            # decode
            z0_est_qi, _ = vqvae.vq(z0_est_i)
            x_pred_i = vqvae.decode(z0_est_qi)

            task_loss = self._expert_task_loss(i, x_pred_i, x_clean)
            expert_losses.append(diff_loss + task_loss)

        exp_losses = torch.stack(expert_losses, dim=0)  # (k,)

        # gating-prob => approximate by mean over batch
        gp_mean = gating_probs.mean(dim=0)
        weighted_expert_loss = (exp_losses * gp_mean).sum()

        # gating supervision
        cos_sims = []
        for i in range(len(eps_preds)):
            cs_i = cosine_similarity_4d(z0_est_list[i], z0_clean)
            cos_sims.append(cs_i)
        cos_sims = torch.stack(cos_sims, dim=1)  # (B,k)

        mad_val = median_absolute_deviation(cos_sims)
        T = 1.4826*mad_val.item()
        if T<1e-8:
            T=1e-8
        gating_gt = F.softmax(cos_sims/T, dim=1)
        ce = - (gating_gt * torch.log(gating_probs+1e-8)).sum(dim=1).mean()

        # gating diversity
        pair_sims = []
        for i in range(len(eps_preds)):
            sims_i = []
            for j in range(len(eps_preds)):
                if j==i: continue
                sij = cosine_similarity_4d(z0_est_list[i], z0_est_list[j])
                sims_i.append(sij.mean())
            pair_sims.append(torch.stack(sims_i).max())
        pair_sims = torch.stack(pair_sims)
        mean_of_max = pair_sims.mean()
        diversity_term = torch.log(1.0 + mean_of_max/len(eps_preds))

        gating_loss = ce + diversity_term

        # uncertainty-based weighting
        old_ema = self.gating_loss_ema.item()
        new_ema = self.ema_decay*old_ema + (1.-self.ema_decay)*gating_loss.item()
        self.gating_loss_ema.fill_(new_ema)

        old_var = self.gating_loss_var_ema.item()
        delta = gating_loss.item()-new_ema
        new_var = self.ema_decay*old_var + (1.-self.ema_decay)*(delta**2)
        self.gating_loss_var_ema.fill_(new_var)
        w = 1.0/(new_var+1e-8)

        total = weighted_expert_loss + w*gating_loss

        return {
            'total': total,
            'expert': weighted_expert_loss,
            'gating': gating_loss,
            'ce': ce,
            'div': diversity_term,
            'w': w
        }
