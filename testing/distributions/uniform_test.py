import pytest
import mxnet as mx
import numpy as np
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.components.distributions import Uniform
from mxfusion.util.testutils import numpy_array_reshape
from mxfusion.util.testutils import MockMXNetRandomGenerator
from scipy.stats import uniform


@pytest.mark.usefixtures("set_seed")
class TestUniformDistribution(object):

    @pytest.mark.parametrize(
        "dtype, low, low_is_samples, high, high_is_samples, rv, rv_is_samples, num_samples", [
            (np.float64, np.random.rand(5, 2), True, np.random.rand(2) + 1, False, np.random.rand(5, 3, 2) + 0.5, True, 5),
            (np.float64, np.random.rand(5, 2), True, np.random.rand(2) + 2, False, np.random.rand(3, 2) + 1, False, 5),
            (np.float64, np.random.rand(2), False, np.random.rand(2) + 2, False, np.random.rand(3, 2) + 1, False, 5),
            (np.float64, np.random.rand(5, 2), True, np.random.rand(5, 3, 2) + 2, True, np.random.rand(5, 3, 2) + 1, True,
             5),
            (np.float32, np.random.rand(5, 2), True, np.random.rand(2) + 2, False, np.random.rand(5, 3, 2) + 1, True, 5),
        ])
    def test_log_pdf(self, dtype, low, low_is_samples, high, high_is_samples, rv, rv_is_samples,
                     num_samples):
        is_samples_any = any([low_is_samples, high_is_samples, rv_is_samples])
        rv_shape = rv.shape[1:] if rv_is_samples else rv.shape
        n_dim = 1 + len(rv.shape) if is_samples_any and not rv_is_samples else len(rv.shape)
        low_np = numpy_array_reshape(low, low_is_samples, n_dim)
        high_np = numpy_array_reshape(high, high_is_samples, n_dim)
        scale_np = high_np - low_np
        rv_np = numpy_array_reshape(rv, rv_is_samples, n_dim)

        # Note uniform.logpdf takes loc and scale, where loc=a and scale=b-a
        log_pdf_np = uniform.logpdf(rv_np, low_np, scale_np)
        var = Uniform.define_variable(shape=rv_shape, dtype=dtype).factor

        low_mx = mx.nd.array(low, dtype=dtype)
        if not low_is_samples:
            low_mx = add_sample_dimension(mx.nd, low_mx)
        high_mx = mx.nd.array(high, dtype=dtype)
        if not high_is_samples:
            high_mx = add_sample_dimension(mx.nd, high_mx)
        rv_mx = mx.nd.array(rv, dtype=dtype)
        if not rv_is_samples:
            rv_mx = add_sample_dimension(mx.nd, rv_mx)
        variables = {var.low.uuid: low_mx, var.high.uuid: high_mx, var.random_variable.uuid: rv_mx}
        log_pdf_rt = var.log_pdf(F=mx.nd, variables=variables)

        assert np.issubdtype(log_pdf_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, log_pdf_rt) == is_samples_any
        if is_samples_any:
            assert get_num_samples(mx.nd, log_pdf_rt) == num_samples
        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(log_pdf_np, log_pdf_rt.asnumpy(), rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "dtype, low, low_is_samples, high, high_is_samples, rv_shape, num_samples", [
            (np.float64, np.random.rand(5, 2), True, np.random.rand(2) + 0.1, False, (3, 2), 5),
            (np.float64, np.random.rand(2), False, np.random.rand(5, 2) + 0.1, True, (3, 2), 5),
            (np.float64, np.random.rand(2), False, np.random.rand(2) + 0.1, False, (3, 2), 5),
            (np.float64, np.random.rand(5, 2), True, np.random.rand(5, 3, 2) + 0.1, True, (3, 2), 5),
            (np.float32, np.random.rand(5, 2), True, np.random.rand(2) + 0.1, False, (3, 2), 5),
        ])
    def test_draw_samples(self, dtype, low, low_is_samples, high,
                          high_is_samples, rv_shape, num_samples):
        n_dim = 1 + len(rv_shape)
        low_np = numpy_array_reshape(low, low_is_samples, n_dim)
        high_np = numpy_array_reshape(high, high_is_samples, n_dim)

        rv_samples_np = np.random.uniform(low=low_np, high=high_np, size=(num_samples,) + rv_shape)

        rand_gen = MockMXNetRandomGenerator(mx.nd.array(rv_samples_np.flatten(), dtype=dtype))

        var = Uniform.define_variable(shape=rv_shape, dtype=dtype, rand_gen=rand_gen).factor
        low_mx = mx.nd.array(low, dtype=dtype)
        if not low_is_samples:
            low_mx = add_sample_dimension(mx.nd, low_mx)
        high_mx = mx.nd.array(high, dtype=dtype)
        if not high_is_samples:
            high_mx = add_sample_dimension(mx.nd, high_mx)
        variables = {var.low.uuid: low_mx, var.high.uuid: high_mx}

        rv_samples_rt = var.draw_samples(F=mx.nd, variables=variables, num_samples=num_samples)

        assert np.issubdtype(rv_samples_rt.dtype, dtype)
        assert is_sampled_array(mx.nd, rv_samples_rt)
        assert get_num_samples(mx.nd, rv_samples_rt) == num_samples

        if np.issubdtype(dtype, np.float64):
            rtol, atol = 1e-7, 1e-10
        else:
            rtol, atol = 1e-4, 1e-5
        assert np.allclose(rv_samples_np, rv_samples_rt.asnumpy(), rtol=rtol, atol=atol)
