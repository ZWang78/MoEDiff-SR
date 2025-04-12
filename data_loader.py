import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MRIDataset(Dataset):
    """
    A PyTorch Dataset to load paired 3T and 7T slices from:
      - ./mnt/3T
      - ./mnt/7T
    Filenames match: e.g. 3T_T1w_100610_0_slice_000.png vs. 7T_T1w_100610_0_slice_000.png
    Resized to 256 x 256
    """

    def __init__(self,
                 root_3T='./mnt/3T',
                 root_7T='./mnt/7T',
                 transform_size=256):
        """
        Args:
            root_3T (str): Path to 3T MRI slices.
            root_7T (str): Path to 7T MRI slices.
            transform_size (int): The resizing dimension for height and width.
        """
        self.root_3T = root_3T
        self.root_7T = root_7T
        self.filenames_3T = sorted(os.listdir(root_3T))

        self.transform = transforms.Compose([
            transforms.Resize((transform_size, transform_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames_3T)

    def __getitem__(self, idx):
        file_3T = self.filenames_3T[idx]
        file_7T = file_3T.replace('3T', '7T')  # match naming

        path_3T = os.path.join(self.root_3T, file_3T)
        path_7T = os.path.join(self.root_7T, file_7T)

        img_3T = Image.open(path_3T).convert('L')
        img_7T = Image.open(path_7T).convert('L')

        # apply transforms
        img_3T_tensor = self.transform(img_3T)  # shape: (1, 256, 256)
        img_7T_tensor = self.transform(img_7T)  # shape: (1, 256, 256)

        # placeholders for gradient/bias correction
        g = torch.zeros_like(img_7T_tensor)  # shape: (1, 256, 256)
        b = torch.zeros_like(img_7T_tensor)  # shape: (1, 256, 256)

        # combine (7T + b + g) => shape: (3,256,256)
        x_7T_input = torch.cat([img_7T_tensor, b, g], dim=0)

        return {
            '3T': img_3T_tensor,
            '7T': img_7T_tensor,
            '7T_input': x_7T_input,
            'filename': file_3T
        }
