import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import MRIDataset
from vqvae import BiggerDeeperVQVAE
from moe_model import MoEModel

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Dataset & DataLoader
    train_dataset = MRIDataset(root_3T='./mnt/3T',
                               root_7T='./mnt/7T',
                               transform_size=256)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2) VQ-VAE
    vqvae = BiggerDeeperVQVAE(in_ch=1, base_ch=128, embed_dim=512, n_embed=1024).to(device)
    vqvae.apply(init_weights)
    vq_opt = optim.AdamW(vqvae.parameters(), lr=1e-6)

    print(">>> Training VQ-VAE ...")
    for epoch in range(5000):
        for i, batch_data in enumerate(train_loader):
            x_7T = batch_data['7T'].to(device)
            x_rec, vq_loss = vqvae(x_7T)
            rec_loss = nn.functional.mse_loss(x_rec, x_7T)
            loss = rec_loss + vq_loss

            vq_opt.zero_grad()
            loss.backward()
            vq_opt.step()

            if (i+1)%10 == 0:
                print(f"[VQ epoch {epoch}] step={i+1}, rec={rec_loss:.4f}, vq={vq_loss:.4f}")

    # (Freeze VQ-VAE or keep training)
    for p in vqvae.parameters():
        p.requires_grad = False

    # 3) MoE model
    moe = MoEModel(timesteps=[500,1000,2000],
                   base_ch=128,
                   k_experts=3,
                   embed_dim=128,
                   gating_heads=4,
                   gating_win=4,
                   gating_depth=2).to(device)
    moe.apply(init_weights)
    moe_opt = optim.Adam(moe.parameters(), lr=1e-4)

    print(">>> Training MoE ...")
    for epoch in range(3000):
        for i, batch_data in enumerate(train_loader):
            x_3T = batch_data['3T'].to(device)         # (B,1,256,256)
            x_7T_input = batch_data['7T_input'].to(device)  # (B,3,256,256)
            # encode => z0
            with torch.no_grad():
                _, z_q, _ = vqvae.encode(x_7T_input)    # shape (B,256,8,8)

            t = torch.randint(0, 50, (z_q.size(0),), device=device, dtype=torch.long)
            out_dict = moe(x_3T, z_q, t)
            loss_dict = moe.compute_loss(out_dict, z_q, vqvae)
            total = loss_dict['total']

            moe_opt.zero_grad()
            total.backward()
            moe_opt.step()

            if (i+1)%10 == 0:
                print(f"[MoE epoch {epoch}] step={i+1}, total={loss_dict['total']:.4f}, "
                      f"expert={loss_dict['expert']:.4f}, gating={loss_dict['gating']:.4f}, "
                      f"ce={loss_dict['ce']:.4f}, div={loss_dict['div']:.4f}, w={loss_dict['w']:.2f}")

    print("Training completed.")

if __name__ == "__main__":
    main()
