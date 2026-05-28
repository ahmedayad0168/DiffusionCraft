import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, image_size=128):
        self.image_folder = image_folder
        self.images = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.exists(os.path.join(image_folder, f.rsplit(".", 1)[0] + ".txt"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)  # -> [-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(os.path.join(self.image_folder, img_name)).convert('RGB')
        img = self.transform(img)

        txt_path = img_name.rsplit('.',1)[0] + '.txt'
        with open(os.path.join(self.image_folder, txt_path), 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        return img, caption
