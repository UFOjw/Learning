import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.png"))
        image = to_tensor(Image.open(img_path).convert("L"))
        mask = to_tensor(Image.open(mask_path).convert("L"))
        mask[mask == 255.0] = 1.0

        return image, mask
