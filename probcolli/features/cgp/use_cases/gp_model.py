from torch import Tensor
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RQKernel
from gpytorch.distributions import MultivariateNormal


class GPModel(ApproximateGP):
    def __init__(self, inducing_points: Tensor, dof: int) -> None:
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module: ZeroMean = ZeroMean()
        self.covar_module: ScaleKernel = ScaleKernel(RQKernel(ard_num_dims=dof))

    def forward(self, x) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
