import torch
from tqdm import tqdm
import numpy as np


class Lick_Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_neurons) -> None:
        super(Lick_Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_neurons)
        self.fc2 = torch.nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = torch.nn.Linear(hidden_neurons, 1)

    def forward(self, x):
        # x = (x.T - x[:,:33].mean(1)).T
        x = torch.nn.ReLU()(self.fc1(x))
        x = torch.nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def prepare_classifier(
    filt_jaw_train,
    filt_jaw_test,
    session_info_train,
    session_info_test,
    device,
    remove_mean=False,
):
    torch.manual_seed(0)
    lick_classifier = Lick_Classifier(filt_jaw_train.shape[0], 128)
    optimizer = torch.optim.AdamW(
        lick_classifier.parameters(), lr=1e-4, weight_decay=0.1
    )
    lick_classifier.to(device)
    num_session = filt_jaw_train.shape[2]
    pbar = tqdm(range(150))
    for epoch in pbar:
        torch.manual_seed(0)
        acc, test_acc, gt = 0, 0, 0
        for session in range(num_session):
            optimizer.zero_grad()
            inputs = filt_jaw_train[:, :, session].T
            trials = ~torch.isnan(inputs.sum(1))
            inputs = inputs[trials]
            if remove_mean:
                mean = inputs[:, :20].mean()
                std = inputs[:, :20].std()
                inputs = (inputs - mean) / std
            labels = torch.tensor(np.isin(session_info_train[0][session], [1, 3])) * 1.0
            labels = labels.to(device)[: trials.sum()]

            input_test = filt_jaw_test[:, :, session].T
            trials = ~torch.isnan(input_test.sum(1))
            input_test = input_test[trials]
            if remove_mean:
                input_test = (input_test - mean) / std
            labels_test = (
                torch.tensor(np.isin(session_info_test[0][session], [1, 3])) * 1.0
            )
            labels_test = labels_test.to(device)[: trials.sum()]

            output = lick_classifier(inputs).flatten()
            loss = torch.nn.BCELoss()(output, labels)
            loss.backward()
            optimizer.step()
            acc += (((output > 0.5) == labels) * 1.0).mean()
            gt += max(labels.mean(), 1 - labels.mean())
            output_test = lick_classifier(input_test).flatten()
            test_acc += (((output_test > 0.5) == labels_test) * 1.0).mean()
        pbar.set_postfix_str(f"test_accuracy {test_acc.item()/num_session:.3f}")

    return lick_classifier
