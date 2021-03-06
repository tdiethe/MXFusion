import unittest
import mxnet as mx
import numpy as np
import mxnet.gluon.nn as nn
import mxfusion as mf
from mxfusion.inference.map import MAP
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.inference import VariationalPosteriorForwardSampling, GradBasedInference


class MAPTests(unittest.TestCase):
    """
    Test class that tests the MXFusion.utils methods.
    """

    def setUp(self):
        self.D = 10
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(nn.Dense(self.D, activation="relu"))
            self.net.add(nn.Dense(1, activation="relu"))
        self.net.initialize()

        from mxnet.gluon import HybridBlock
        class DotProduct(HybridBlock):
            def hybrid_forward(self, F, x, *args, **kwargs):
                return F.dot(x, args[0])
        self.mx_dot = DotProduct()

        m = mf.models.Model()
        m.mean = mf.components.Variable()
        m.var = mf.components.Variable(transformation=PositiveTransformation())
        m.N = mf.components.Variable()
        m.x = mf.components.distributions.Normal.define_variable(mean=m.mean, variance=m.var, shape=(m.N,))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.x, variance=mx.nd.array([1]), shape=(m.N,))
        self.m = m

        q = mf.models.posterior.Posterior(m)
        q.a = mf.components.Variable()
        q.x.set_prior(mf.components.distributions.PointMass(location=q.a))
        self.q = q


        m = mf.models.Model()
        m.mean = mf.components.Variable()
        m.var = mf.components.Variable(transformation=PositiveTransformation())
        m.N = mf.components.Variable()
        m.x = mf.components.distributions.Normal.define_variable(mean=m.mean, variance=m.var, shape=(m.N,))
        m.y = mf.components.distributions.Normal.define_variable(mean=m.x, variance=mx.nd.array([1]), shape=(m.N,))
        self.m2 = m

        q = mf.models.posterior.Posterior(m)
        q.a = mf.components.Variable()
        q.x.set_prior(mf.components.distributions.PointMass(location=q.a))
        self.q2  = q

    def test_one_map_example(self):
        """
        Tests that the creation of variables from a base gluon block works correctly.
        """
        from mxfusion.inference.map import MAP
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [self.m.y]
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.run(y=mx.nd.array(np.random.rand(10)), max_iter=10)

    def test_function_map_example(self):
        """
        Tests that the creation of variables from a base gluon block works correctly.
        """
        from mxfusion.inference.map import MAP
        from mxfusion.inference.grad_based_inference import GradBasedInference
        from mxfusion.inference import BatchInferenceLoop
        observed = [self.m.y, self.m.x]
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg, grad_loop=BatchInferenceLoop())
        infr.run(y=mx.nd.array(np.random.rand(self.D)), x=mx.nd.array(np.random.rand(self.D)), max_iter=10)

    def test_inference_outcome_passing_success(self):
        observed = [self.m.y, self.m.x]
        alg = MAP(model=self.m, observed=observed)
        infr = GradBasedInference(inference_algorithm=alg)
        infr.run(y=mx.nd.array(np.random.rand(self.D)),
                 x=mx.nd.array(np.random.rand(self.D)), max_iter=1)

        infr2 = VariationalPosteriorForwardSampling(10, [self.m.x], infr, [self.m.y])
        infr2.run(x=mx.nd.array(np.random.rand(self.D)))

        # infr2 = mf.inference.MAPInference(model_graph=self.m2, post_graph=self.q2, observed=[self.m2.y, self.m2.x], hybridize=False)
        # infr2.run(y=mx.nd.array(np.random.rand(1)),
        #           x=mx.nd.array(np.random.rand(self.D)),
        #           inference_outcomes=infr.params,
        #           var_ties={self.q2.a.uuid: self.q.a.uuid})
        #
        # assert infr.params.param_dict[self.q.a] == infr2.params.param_dict[self.q.a]
