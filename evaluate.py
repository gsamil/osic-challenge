import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import CNN
from dataset import OSICDataset


def load_checkpoint(model_path, device):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    return model


def get_best_checkpoint_path(model_dir):
    epochs = []
    model_names = []
    for model_name in os.listdir(model_dir):
        epoch = int(model_name.split(".")[0].split("_")[1])
        epochs.append(epoch)
        model_names.append(model_name)
    max_index = np.argmax(epochs)
    return os.path.join(model_dir, model_names[max_index])


if __name__ == '__main__':
    batch_size = 1
    num_workers = 0
    data_dir = r'C:\Users\abdullah\Desktop\projects\osic-pulmonary-fibrosis-progression\data'
    checkpoint_path = get_best_checkpoint_path("../result_00")
    test_csv_file_path = os.path.join(data_dir, 'test.csv')
    test_data_dir = os.path.join(data_dir, 'test')
    submission_path = '../submission.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))

    test_dataset = OSICDataset(csv_file_path=test_csv_file_path, data_dir=test_data_dir, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = load_checkpoint(checkpoint_path, device)

    data_test = {"Patient_Week": [], "FVC": [], "Confidence": []}

    model.eval()
    with torch.no_grad():
        for eidx, (sample_batch) in enumerate(test_dataloader):
            patient_batch = sample_batch['patient']
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            fvc_preds, typical_fvc_preds = model(image_batch, scalars_batch)
            fvc_predictions = fvc_preds.data.cpu().detach().numpy()
            typical_fvc_predictions = typical_fvc_preds.data.cpu().detach().numpy()
            labels = labels_batch.cpu().detach().numpy()
            fvc_labels = (labels[:, 0] * (6399 - 827)) + 827
            fvc_predictions = (fvc_predictions * (6399 - 827)) + 827
            typical_fvc_predictions = (typical_fvc_predictions * (6399 - 827)) + 827
            confidence_predictions = np.abs(fvc_predictions - typical_fvc_predictions)
            fvc_predictions = [int(round(f)) for f in fvc_predictions]
            typical_fvc_predictions = [int(round(f)) for f in typical_fvc_predictions]
            confidence_predictions = [int(round(c)) for c in confidence_predictions]

            weeks = scalars_batch.cpu().detach().numpy()
            weeks = [int(round(w)) for w in (weeks[:, 0] * 145) - 12]

            for patient, week, fvc, confidence in zip(patient_batch, weeks, fvc_predictions, confidence_predictions):
                data_test["Patient_Week"].append("{}_{:d}".format(patient, week))
                data_test["FVC"].append(fvc)
                data_test["Confidence"].append(confidence)

            test_df = pd.DataFrame(data_test)
            test_df.to_csv(submission_path, index=False, header=True)

    print("Evaluation finished")

