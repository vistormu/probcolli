import torch
import numpy as np
import torchbnn as bnn

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


from ...core import Logger
from .entities import CBNNInfo


class CBNN:
    def __init__(self, dof: int) -> None:
        self.model: nn.Sequential = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=dof, out_features=50),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=50, out_features=30),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=30, out_features=10),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=10, out_features=1),
            nn.Sigmoid(),
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self,
              input_data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 10,
              batch_size: int = 256,
              lr: float = 0.01,
              ) -> None:

        # Function requirements
        if not np.all(np.logical_and(input_data >= -1, input_data <= 1)):
            raise ValueError('input data must be in the interval [-1, 1]')

        if not np.all(np.logical_or(labels == 0, labels == 1)):
            raise ValueError('labels must be either 0 or 1')

        # Tensor initialization
        train_x: Tensor = Tensor(input_data).cuda() if torch.cuda.is_available() else Tensor(input_data)
        train_y: Tensor = Tensor(labels).cuda() if torch.cuda.is_available() else Tensor(labels)

        # Set into training mode
        self.model.train()

        # Optimizers
        ce_loss = nn.BCELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.1

        optimizer = Adam(self.model.parameters(), lr=lr)

        # Training batch
        dataset: TensorDataset = TensorDataset(train_x, train_y)
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(epochs):
            for x_batch, y_batch in data_loader:

                pre = self.model(x_batch)
                ce: Tensor = ce_loss(pre[..., 0], y_batch)
                kl: Tensor = kl_loss(self.model)
                cost: Tensor = ce + kl_weight*kl

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

            Logger.info(f'Training in progress... {int((i/epochs)*100)}%', flush=True)

    def predict(self, input_data: np.ndarray, beta: float = 0.5, samples_test: int = 10) -> CBNNInfo:
        # TODO: check length of the input data
        # Initialize tensors
        test_x: Tensor = Tensor(input_data).cuda() if torch.cuda.is_available() else Tensor(input_data)

        samples = []

        for i in range(samples_test):

            predicted = self.model(test_x)
            predicted = predicted.detach().cpu().numpy()
            samples.append(predicted)

        mean = np.reshape(np.mean(samples, axis=0), (test_x.size()[0],))
        deviation = np.reshape(np.std(samples, axis=0), (test_x.size()[0],))
        variance = np.power(deviation, 2)

        decision: np.ndarray = mean + beta*deviation

        return CBNNInfo(decision=decision,
                        mean=mean,
                        deviation=deviation,
                        variance=variance,
                        )

    def save(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename+'_model.pth')

    def load(self, filename: str) -> None:
        # TODO: check that dof matches with model
        state_dict = torch.load(filename+'_model.pth')
        self.model.load_state_dict(state_dict)
