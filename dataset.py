import os
import torch
import numpy as np
import pydicom
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset


def check_dataframe():
    train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
    print("train_df :")
    print(train_df.head())
    train_df["Weeks"] = (train_df["Weeks"] + 12) / 145
    train_df["Percent"] = train_df["Percent"] / 100
    train_df["Age"] = train_df["Age"] / 100
    train_df["SmokingStatus"].replace({"Currently smokes": 0.0, "Ex-smoker": 0.5, "Never smoked": 1.0}, inplace=True)
    print(train_df.head())

    unique_patient_df = train_df.drop(['Weeks', 'FVC', 'Percent'], axis=1).drop_duplicates().reset_index(drop=True)
    unique_patient_df['# visits'] = [train_df['Patient'].value_counts().loc[pid] for pid in
                                     unique_patient_df['Patient']]
    print("\nunique_patient_df :")
    print(unique_patient_df.head())
    print(unique_patient_df.iloc[0])

    print('Number of data points: ' + str(len(train_df)))
    print('----------------------')

    for col in train_df.columns:
        print('{} : {} unique values, {} missing.'.format(col,
                                                          len(train_df[col].unique()),
                                                          train_df[col].isna().sum()))

    col = "# visits"
    print('{} : {} unique values, {} missing.'.format(col,
                                                      len(unique_patient_df[col].unique()),
                                                      unique_patient_df[col].isna().sum()))
    print('----------------------')


class OSICDataset(Dataset):
    def __init__(self, csv_file_path, data_dir, transform=None, is_test=False):
        """
        Args:
            csv_file_path (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.is_test = is_test
        self.data_dir = data_dir
        self.transform = transform
        self.df = self._convert_df(pd.read_csv(csv_file_path))
        self.image_dict = self._load_image_dict(self.df)
        self.df = self.df.loc[self.df["Patient"].isin(self.image_dict)]
        if self.is_test:
            self.df = self._make_test_dataframe(self.df)
        print("All images are read.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient, weeks, fvc, percent, age, sex, smoking_status = self.df.iloc[idx]

        if patient not in self.image_dict:
            return None

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
            scans = load_scan(patient_dir)
            image = get_pixels_hu(scans)
            if len(image) == 0:
                continue
            image = torch.Tensor(np.stack(image))
            image = F.interpolate(image[None, None, :], size=(64, 64, 64))[0, 0, :]
            image_dict[patient] = image
        return image_dict

    def _make_test_dataframe(self, df):
        data_test = {"Patient": [], "Weeks": [], "FVC": [], "Percent": [], "Age": [], "Sex": [], "SmokingStatus": []}
        for i in range(-12, 134):
            for index, row in df.iterrows():
                data_test["Patient"].append(row["Patient"])
                data_test["Weeks"].append(str((i + 12) / 145))
                data_test["FVC"].append(row["FVC"])
                data_test["Percent"].append(row["Percent"])
                data_test["Age"].append(row["Age"])
                data_test["Sex"].append(row["Sex"])
                data_test["SmokingStatus"].append(row["SmokingStatus"])
        return pd.DataFrame(data_test)


def load_scan(path):
    """
    Loads scans from a folder and into a list.
    Parameters: path (Folder path)
    Returns: slices (List of slices)
    """
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        try:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        except:
            slice_thickness = None
    if slice_thickness is not None:
        for s in slices:
            s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(scans):
    """
    Converts raw images to Hounsfield Units (HU).
    Parameters: scans (Raw images)
    Returns: image (NumPy array)
    """
    image = []
    for s in scans:
        try:
            image.append(s.pixel_array)
        except:
            print("read error.")
    if len(image) == 0:
        return []
    image = np.stack(image)
    image = image.astype(np.int16)
    # Since the scanning equipment is cylindrical in nature and image output is square, we set the out-of-scan pixels to 0
    image[image == -2000] = 0
    # HU = m*P + b
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)
