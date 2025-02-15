from torch.utils.data import Dataset
import numpy as np
import pydicom
import os


class RSNADataLoader(Dataset):
    def __init__(self, surveys_df, image_dir, transform=None):
        self.image_dir = image_dir
        self.surveys_df = surveys_df
        self.transform = transform

    def __getitem__(self, args):
        if isinstance(args, tuple):
            if len(args) == 2:
                status, column = args
                return self._get_random_item(status, column)
            else:
                raise ValueError("Expected two arguments for 'status' and 'column'")
        else:
            idx = args
            return self._get_index_item(idx)

    def _get_index_item(self, idx):
        survey = self.surveys_df.iloc[idx]
        study_id = survey.iloc[0]
        series_id = survey.iloc[1]
        instance_number = str(survey.iloc[2]) + ".dcm"
        image_path = os.path.join(self.image_dir, str(study_id), str(series_id), instance_number)
        dcm_img = pydicom.dcmread(image_path).pixel_array.astype(np.float32)

        if self.transform:
            dcm_img = self.transform(dcm_img)

        sample = {"image": dcm_img, "labels": survey[3:].values}
        return sample

    def _get_random_item(self, status, column):
        survey = self.surveys_df[['study_id', 'series_id', 'instance_number', column]]
        survey = survey[survey[column] == status].sample()
        study_id = survey.iloc[0, 0]
        series_id = survey.iloc[0, 1]
        instance_number = str(survey.iloc[0, 2]) + ".dcm"
        image_path = os.path.join(self.image_dir, str(study_id), str(series_id), instance_number)
        dcm_img = pydicom.dcmread(image_path).pixel_array

        if self.transform:
            dcm_img = self.transform(dcm_img)

        sample = {"image": dcm_img, "labels": survey.iloc[0, 3].astype(np.int8)}
        return sample

    def __len__(self):
        return len(self.surveys_df)
