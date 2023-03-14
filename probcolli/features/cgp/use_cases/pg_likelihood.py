import torch

from torch.distributions import Bernoulli
from gpytorch.likelihoods import _OneDimensionalLikelihood


class PGLikelihood(_OneDimensionalLikelihood):
    def expected_log_prob(self, target, input, *args, **kwargs):
        mean, variance = input.mean, input.variance
        raw_second_moment = variance + mean.pow(2)
        target = target.to(mean.dtype).mul(2.).sub(1.)
        c = raw_second_moment.detach().sqrt()
        half_omega = 0.25 * torch.tanh(0.5 * c) / c
        res = 0.5 * target * mean - half_omega * raw_second_moment
        res = res.sum(dim=-1)

        return res

    def forward(self, function_samples):
        return Bernoulli(logits=function_samples)

    def marginal(self, function_dist):
        def prob_lambda(function_samples): return self.forward(function_samples).probs
        probs = self.quadrature(prob_lambda, function_dist)
        return Bernoulli(probs=probs)
