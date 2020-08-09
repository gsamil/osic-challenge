import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CNN
from dataset import OSICDataset


def make_test_file(test_csv_path, out_csv_path):
    test_csv = open(test_csv_path, "rt", encoding="utf-8")
    test_lines = [l.replace("\n", "") for l in test_csv.readlines()[1:]]
    test_csv.close()
    out_csv = open(out_csv_path, "wt", encoding="utf-8")
    out_csv.write("Patient,Weeks,FVC,Percent,Age,Sex,SmokingStatus\n")
    for i in range(-12, 134):
        for line in test_lines:
            Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus = line.split(",")
            out_csv.write(",".join([Patient, str(i), "2000", "100", Age, Sex, SmokingStatus]) + "\n")
    out_csv.close()


def load_checkpoint(model_path, device):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
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
    eval_csv_file_path = os.path.join(data_dir, 'test_all.csv')
    test_data_dir = os.path.join(data_dir, 'test')
    submission_path = '../submission.csv'

    make_test_file(test_csv_file_path, eval_csv_file_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))

    test_dataset = OSICDataset(csv_file_path=eval_csv_file_path, data_dir=test_data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = load_checkpoint(checkpoint_path, device)

    logger = open(submission_path, "wt", encoding="utf-8")
    logger.write("Patient_Week,FVC,Confidence\n")

    all_predictions = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for eidx, (sample_batch) in enumerate(test_dataloader):
            patient_batch = sample_batch['patient']
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            outputs = model(image_batch, scalars_batch)
            predictions = outputs.data.cpu().detach().numpy()
            labels = labels_batch.cpu().detach().numpy()
            fvc_labels = (labels[:, 0] * (6399 - 827)) + 827
            fvc_predictions = [int(f) for f in (predictions[:, 0] * (6399 - 827)) + 827]
            confidence_labels = labels[:, 1]*100
            confidence_predictions = [int(p) for p in predictions[:, 1]*100]
            weeks = scalars_batch.cpu().detach().numpy()
            weeks = [int(w) for w in (weeks[:, 0] * 145) - 12]

            for patient, week, fvc, confidence in zip(patient_batch, weeks, fvc_predictions, confidence_predictions):
                logger.write("{}_{:d},{:d},{:d}\n".format(patient, week, fvc, confidence))

    logger.close()
    print("Evaluation finished")

