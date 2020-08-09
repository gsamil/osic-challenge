import os
import torch
from dataset import OSICDataset
from torch.utils.data import DataLoader
from train import get_logger, get_predictions


def load_checkpoint(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


if __name__ == '__main__':
    batch_size = 1
    num_workers = 0
    data_dir = r'C:\Users\abdullah\Desktop\projects\osic-pulmonary-fibrosis-progression\data'
    checkpoint_path = "../result_00/model_070.pth"
    test_csv_file_path = os.path.join(data_dir, 'test_all.csv')
    test_data_dir = os.path.join(data_dir, 'test')
    submission_path = '../submission.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))

    test_dataset = OSICDataset(csv_file_path=test_csv_file_path, data_dir=test_data_dir)
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
