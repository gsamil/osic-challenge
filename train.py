import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
from model import CNN
from loss import laplace_log_likelihood, laplace_log_likelihood_loss
from dataset import OSICDataset


def get_logger(log_path=None):
    log_level = logging.INFO
    log = logging.getLogger('osic-pulmonary-fibrosis-progression')
    log.setLevel(log_level)
    log.addHandler(logging.StreamHandler())
    if log_path is not None and log_path is not "":
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path, 'wt')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def get_predictions(model, data_loader, device, logger=None):
    all_predictions = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for eidx, (sample_batch) in enumerate(data_loader):
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            outputs = model(image_batch, scalars_batch)
            all_predictions.extend(outputs.data.cpu().detach().numpy())
            all_labels.extend(labels_batch.cpu().detach().numpy())
    return np.asarray(all_labels), np.asarray(all_predictions)


if __name__ == '__main__':
    max_epoch = 100
    patience = 50
    batch_size = 1
    num_workers = 0
    learning_rate = 1e-3
    save_limit = 5
    data_dir = r'C:\Users\abdullah\Desktop\projects\osic-pulmonary-fibrosis-progression\data'
    model_dir = "../result_00"
    train_csv_file_path = os.path.join(data_dir, 'train.csv')
    test_csv_file_path = os.path.join(data_dir, 'test.csv')
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')
    log_path = '../log_train_00.txt'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = get_logger(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : {}".format(device))

    train_dataset = OSICDataset(csv_file_path=train_csv_file_path, data_dir=train_data_dir)
    test_dataset = OSICDataset(csv_file_path=test_csv_file_path, data_dir=test_data_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = CNN()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3)

    bad_counter = 0
    history_errs = [-1000.0]
    saved_models = []

    logger.info("Training started.")
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        model.train()
        train_predictions = []
        train_labels = []
        for sample_batch in train_dataloader:
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            optimizer.zero_grad()
            outputs = model(image_batch, scalars_batch)
            # loss = criterion(outputs, labels_batch)
            loss = laplace_log_likelihood_loss(labels_batch[:, 0], outputs[:, 0], outputs[:, 1])
            print(loss.item())
            train_predictions.extend(outputs.data.cpu().detach().numpy())
            train_labels.extend(labels_batch.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        train_labels = np.asarray(train_labels)
        train_predictions = np.asarray(train_predictions)
        train_metric = laplace_log_likelihood(train_labels[:, 0], train_predictions[:, 0], train_predictions[:, 1])
        test_labels, test_predictions = get_predictions(model, test_dataloader, device)
        test_metric = laplace_log_likelihood(test_labels[:, 0], test_predictions[:, 0], test_predictions[:, 1])

        logger.info('Epoch %d Train %s: %f, Test %s: %f, Loss: %f' % (
            epoch,
            "LLL",
            train_metric,
            "LLL",
            test_metric,
            loss.item()
        ))

        if epoch == 0 or test_metric > np.max(history_errs):
            bad_counter = 0
            best_epoch_num = epoch
            logger.info("Saving current best model at epoch %d", best_epoch_num)
            save_path = os.path.join(model_dir, "model_{:03d}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)
            saved_models.append(save_path)
            if len(saved_models) > save_limit:
                os.remove(saved_models[0])
                saved_models = saved_models[1:]

        if test_metric <= np.max(history_errs):
            bad_counter += 1
            logger.info("Bad counter: %s", bad_counter)

        history_errs.append(test_metric)

        if bad_counter > patience:
            logger.info("Early stop at epoch %d" % epoch)
            break

    logger.info('Finished Training')
