import torch
import numpy as np

from vclog import Logger
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli
from gpytorch.optim import NGD
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from .use_cases import GPModel, PGLikelihood
from .entities import CGPInfo


class CGP:
    def __init__(self, inducing_points: int, dof: int) -> None:
        self.inducing_points: int = inducing_points
        self.dof: int = dof
        self.model: GPModel = GPModel(Tensor(np.random.uniform(-1, 1, (inducing_points, dof))), dof)
        self.likelihood: PGLikelihood = PGLikelihood()

        self._logger: Logger = Logger('cgp')

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def train(self,
              input_data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 10,
              batch_size: int = 256,
              variational_lr: float = 0.5,
              hyperparameter_lr: float = 0.1,
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
        self.likelihood.train()

        # Optimizers
        variational_ngd_optimizer: NGD = NGD(self.model.variational_parameters(), num_data=train_y.size(0), lr=variational_lr)
        hyperparameter_optimizer: Adam = Adam([{'params': self.model.hyperparameters(), 'params': self.likelihood.parameters()},], lr=hyperparameter_lr)
        loss_function: VariationalELBO = VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        # Training batch
        dataset: TensorDataset = TensorDataset(train_x, train_y)
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i in range(epochs):
            for j, (x_batch, y_batch) in enumerate(data_loader):
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -loss_function(output, y_batch)  # type: ignore
                loss.backward()

                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

                self._logger.info(f'Training in progress... {int((i/(epochs+1))*100+(j/len(data_loader))*100/epochs)}%', flush=True)

        self._logger.info('Training in progress... 100%')

    def predict(self, input_data: np.ndarray, beta: float = 0.5) -> CGPInfo:
        if not input_data.size - len(input_data):
            input_data = np.array([input_data])

        # Initialize tensors
        test_x: torch.Tensor = torch.Tensor(input_data).cuda() if torch.cuda.is_available() else torch.Tensor(input_data)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            predictions: Bernoulli = self.likelihood(self.model(test_x))  # type:ignore
            mean: np.ndarray = predictions.mean.cpu().numpy()  # type:ignore
            deviation: np.ndarray = predictions.stddev.cpu().numpy()  # type: ignore
            variance: np.ndarray = predictions.variance.cpu().numpy()  # type: ignore

        decision: np.ndarray = mean + beta*deviation

        return CGPInfo(decision=decision,
                       mean=mean,
                       deviation=deviation,
                       variance=variance,
                       )

    def save(self, destination: str) -> None:
        torch.save(self.model.state_dict(), destination+'model.pth')
        torch.save(self.likelihood.state_dict(), destination+'likelihood.pth')
        np.savetxt(destination+'class_data.txt', [self.inducing_points, self.dof], delimiter=',')

    @staticmethod
    def from_model(directory: str):
        inducing_points, dof = np.loadtxt(directory+'class_data.txt')
        cgp = CGP(int(inducing_points), int(dof))
        cgp.model.load_state_dict(torch.load(directory+'model.pth', map_location=torch.device('cpu')))
        cgp.likelihood.load_state_dict(torch.load(directory+'likelihood.pth', map_location=torch.device('cpu')))

        return cgp
