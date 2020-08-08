import os
import torch
import numpy as np
import pydicom
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset


class OSICDataset(Dataset):
    def __init__(self, csv_file_path, data_dir, transform=None):
        """
        Args:
            csv_file_path (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = self._convert_df(pd.read_csv(csv_file_path))
        self.data_dir = data_dir
        self.transform = transform
        self.image_dict = self._load_image_dict(self.df)
        print("All images are read.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient, weeks, fvc, percent, age, sex, smoking_status = self.df.iloc[idx]
        image = self.image_dict[patient]

        sample = {
            'patient': patient,
            'image': image,
            'scalars': np.asarray([weeks, age, sex, smoking_status]).astype(np.float64),
            'labels': np.asarray([fvc, percent]).astype(np.float64)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _convert_df(self, df):
        df["Weeks"] = (df["Weeks"] + 12) / 145
        df["Percent"] = df["Percent"] / 100
        df["Age"] = df["Age"] / 100
        df["SmokingStatus"].replace({"Currently smokes": 0.0, "Ex-smoker": 0.5, "Never smoked": 1.0}, inplace=True)
        df["Sex"].replace({"Male": 0.0, "Female": 1.0}, inplace=True)
        df['FVC'] = (df['FVC'] - 827) / (6399 - 827)
        return df

    def _load_image_dict(self, df):
        image_dict = {}
        for patient in df["Patient"]:
            if patient in image_dict:
                continue
            patient_dir = os.path.join(self.data_dir, patient)
            image = []
            for dcm_name in os.listdir(patient_dir):
                try:
                    dcm_path = os.path.join(patient_dir, dcm_name)
                    dcm_file = pydicom.dcmread(dcm_path)
                    dcm_array = np.asarray(dcm_file.pixel_array.astype(np.float64))
                    image.append(dcm_array)
                except Exception as e:
                    print("{} {}".format(patient, dcm_name))
                    raise
            image = torch.Tensor(np.stack(image))
            image = F.interpolate(image[None, None, :], size=(64, 64, 64))[0, 0, :]
            image_dict[patient] = image
        return image_dict