from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2
import torch
from torchvision import transforms

class MuraDataset(Dataset):
    def __init__(self, is_training, dir_path = './dataset/MURA-v1.1/',transform = None,
    regions = ['ELBOW',"FINGER","FOREARM","HAND","HUMERUS","SHOULDER","WRIST"]
    ):
        self.dataset_dir = Path(dir_path)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_dir}") 

        df_img= pd.read_csv(self.dataset_dir / 'train_image_paths.csv', header=None, names=['image_path'])
        df_labels = pd.read_csv(self.dataset_dir / 'train_labeled_studies.csv', header=None, names=['study_dir','label'])

        # Capture up to and including study dir with trailing slash
        pat = r"^(.*?/study\d+_(?:positive|negative)/)"
        df_img["study_dir"] = df_img["image_path"].str.extract(pat, expand=False)

        df = df_img.merge(df_labels, on="study_dir", how="inner")
        df['label'] = df['label'].astype(int)

        df["region"] = df["image_path"].str.split("/").str[2].str.replace("^XR_", "", regex=True)
        if regions:
            df = df[df["region"].isin(regions)]

        self.samples = df.reset_index(drop=True)

        if transform is None:
            if is_training:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        image_path = self.dataset_dir.parent / self.samples.iloc[index]['image_path']
        img = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
        if img is None: 
            raise FileNotFoundError(f"Could not load image at: {str(image_path)}")

        label = self.samples.iloc[index]['label']
        img_tensor = self.transform(img)

        return img_tensor, torch.tensor(label, dtype=torch.long)



