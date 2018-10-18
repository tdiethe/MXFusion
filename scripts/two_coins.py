import pytest
import mxnet as mx
import numpy as np
from scipy.stats import bernoulli

from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Bernoulli
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator


@pytest.mark.parametrize(
    "dtype", [
        # np.float64,
        np.float32,
    ])
def test_two_coins(self, dtype):
    from mxfusion.inference.map import MAP
    from mxfusion.inference import StochasticVariationalInference
    from mxfusion.inference.grad_based_inference import GradBasedInference
    from mxfusion.inference import BatchInferenceLoop
    from mxfusion.models import Model, Posterior
    from mxfusion.components.variables import Variable, VariableType

    # perform inference over both_heads variable
    m = Model()
    prob_true = mx.nd.array([0.5], dtype=dtype)
    m.coin1 = Bernoulli.define_variable(prob_true, shape=(1,), rand_gen=None, dtype=dtype)
    m.coin2 = Bernoulli.define_variable(prob_true, shape=(1,), rand_gen=None, dtype=dtype)
    m.both_heads = m.coin1 * m.coin2

    q = Posterior(m)
    for v in m.variables.values():
        if v.type == VariableType.RANDVAR:
            q[v].set_prior(Bernoulli(prob_true=Variable(shape=v.shape), dtype=dtype))

    # # Forwards inference
    # alg = StochasticVariationalInference(model=m, observed=[], posterior=q, num_samples=1)
    # inf = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop(), dtype=dtype)
    # inf.run(max_iter=10)

    # Look at value of both_heads variable

    # Going backwards
    observed = [m.both_heads]
    # alg = MAP(model=m, observed=observed)
    alg = StochasticVariationalInference(model=m, observed=observed, posterior=q, num_samples=10)
    infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
    infr.run(both_heads=mx.nd.array([False]), max_iter=10)

    # Look at value of coin1
    coin1_post = infr.params[q.coin1.factor.prob_true].asnumpy()[0]
    assert np.allclose(coin1_post, 1.0 / 3)
