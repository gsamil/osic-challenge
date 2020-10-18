import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
from model import CNN, FFNet
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
    all_fvc_predictions = []
    all_typical_fvc_predictions = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for eidx, (sample_batch) in enumerate(data_loader):
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            fvc_preds, typical_fvc_preds = model(image_batch, scalars_batch)
            all_fvc_predictions.extend(fvc_preds.data.cpu().detach().numpy())
            all_typical_fvc_predictions.extend(typical_fvc_preds.data.cpu().detach().numpy())
            all_labels.extend(labels_batch.cpu().detach().numpy())
    return np.asarray(all_labels), np.asarray(all_fvc_predictions), np.asarray(all_typical_fvc_predictions)


def clean_model_dir(model_dir):
    if os.path.exists(model_dir):
        for model_name in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, model_name))


if __name__ == '__main__':
    max_epoch = 100
    patience = 50
    batch_size = 1
    num_workers = 0
    learning_rate = 1e-3
    save_limit = 5
    data_dir = '../input/osic-pulmonary-fibrosis-progression'
    model_dir = "./models"
    train_csv_file_path = os.path.join(data_dir, 'train.csv')
    test_csv_file_path = os.path.join(data_dir, 'test.csv')
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')
    log_path = '../log_train.txt'

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
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=1e-3)
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=1e-3, initial_accumulator_value=0, eps=1e-10)


    bad_counter = 0
    history_errs = [-1000.0]
    saved_models = []

    logger.info("Training started.")
    for epoch in range(max_epoch):  # loop over the dataset multiple times
        model.train()
        train_fvc_predictions = []
        train_typical_fvc_predictions = []
        train_labels = []
        for sample_batch in train_dataloader:
            image_batch = sample_batch['image'].float().to(device)
            scalars_batch = sample_batch['scalars'].float().to(device)
            labels_batch = sample_batch['labels'].float().to(device)
            optimizer.zero_grad()
            fvc_preds, typical_fvc_preds = model(image_batch, scalars_batch)
            loss1 = criterion(fvc_preds, labels_batch[:, 0])
            loss2 = criterion(typical_fvc_preds, labels_batch[:, 2])
            # loss = laplace_log_likelihood_loss(labels_batch[:, 0], fvc_preds, typical_fvc_preds, device)
            train_fvc_predictions.extend(fvc_preds.data.cpu().detach().numpy())
            train_typical_fvc_predictions.extend(typical_fvc_preds.data.cpu().detach().numpy())
            train_labels.extend(labels_batch.cpu().detach().numpy())
            # loss = loss1 + loss2
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
        train_labels = np.asarray(train_labels)
        train_fvc_predictions = np.asarray(train_fvc_predictions)
        train_typical_fvc_predictions = np.asarray(train_typical_fvc_predictions)
        train_metric = laplace_log_likelihood(train_labels[:, 0], train_fvc_predictions, train_typical_fvc_predictions)
        test_labels, test_fvc_predictions, test_typical_fvc_predictions = get_predictions(model, test_dataloader, device)
        test_metric = laplace_log_likelihood(test_labels[:, 0], test_fvc_predictions, test_typical_fvc_predictions)

        logger.info('Epoch %d Train %s: %f, Test %s: %f, Loss1: %f, Loss2: %f' % (
            epoch,
            "LLL",
            train_metric,
            "LLL",
            test_metric,
            loss1.item(),
            loss2.item()
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
