from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pywt


class WaveletTransform:
    """Custom wavelet transform for texture analysis"""
    def __init__(self, wavelet='db4', levels=3):
        self.wavelet = wavelet
        self.levels = levels

    def __call__(self, image):
        """
        Apply wavelet transform to image and return multi-scale features
        Args:
            image: PIL Image or numpy array
        Returns:
            torch.Tensor: Wavelet coefficients as additional channels
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale for wavelet analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Normalize to [0, 1] for wavelet transform
        gray = gray.astype(np.float32) / 255.0

        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(gray, self.wavelet, level=self.levels)

        # Extract approximation and detail coefficients
        # coeffs[0] = approximation (low-frequency)
        # coeffs[1:] = details (high-frequency) at different scales

        wavelet_features = []

        # Add approximation coefficient (resized to original size)
        approx = coeffs[0]
        approx_resized = cv2.resize(approx, (image.shape[1], image.shape[0]))
        wavelet_features.append(approx_resized)

        # Add detail coefficients from different scales
        for i in range(1, min(len(coeffs), self.levels + 1)):
            # Each detail level has 3 orientations: horizontal, vertical, diagonal
            cH, cV, cD = coeffs[i]

            # Resize to original image size
            cH_resized = cv2.resize(cH, (image.shape[1], image.shape[0]))
            cV_resized = cv2.resize(cV, (image.shape[1], image.shape[0]))
            cD_resized = cv2.resize(cD, (image.shape[1], image.shape[0]))

            wavelet_features.extend([cH_resized, cV_resized, cD_resized])

        # Stack as additional channels (limit to first 4 as you suggested)
        wavelet_stack = np.stack(wavelet_features[:4], axis=2)

        # Normalize wavelet coefficients
        wavelet_stack = (wavelet_stack - wavelet_stack.mean()) / (wavelet_stack.std() + 1e-8)

        # Convert to tensor and ensure proper shape [C, H, W]
        wavelet_tensor = torch.from_numpy(wavelet_stack).permute(2, 0, 1).float()

        return wavelet_tensor


class MuraDataset(Dataset):
    def __init__(self, is_training, dir_path='./dataset/MURA-v1.1/', transform=None,
                 regions=['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST'],
                 use_wavelets=False):
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
        self.use_wavelets = use_wavelets
        self.wavelet_transform = WaveletTransform(wavelet='db4', levels=3) if use_wavelets else None

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
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at: {str(image_path)}")

        label = self.samples.iloc[index]['label']

        # Apply wavelet transform to raw image before other transforms
        if self.use_wavelets:
            wavelet_features = self.wavelet_transform(img)

        img_tensor = self.transform(img)

        if self.use_wavelets:
            # Resize wavelet features to match transformed image size
            wavelet_features = torch.nn.functional.interpolate(
                wavelet_features.unsqueeze(0),
                size=(img_tensor.shape[1], img_tensor.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            # Concatenate RGB channels with wavelet features -> [7, H, W]
            img_tensor = torch.cat([img_tensor, wavelet_features], dim=0)

        return img_tensor, torch.tensor(label, dtype=torch.long)



