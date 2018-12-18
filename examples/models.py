from mxfusion import Variable
from mxfusion.components.distributions import Normal, Beta

import mxnet as mx

from mxfusion.components.functions.operators import transpose
from mxfusion.components.variables.var_trans import Logistic, PositiveTransformation


class BetaIRT:

    def __init__(self, M, C, theta_prior, delta_prior, a_prior):
        self.M = M  # number of instances
        self.C = C  # number of classes
        self.theta_prior = theta_prior  # prior of ability
        self.delta_prior = delta_prior  # prior of difficulty
        self.a_prior = a_prior  # prior of discrimination

        if isinstance(a_prior, Variable):
            # variational posterior of discrimination
            self.qa = Normal.define_variable(
                mean=mx.random.normal(shape=M),
                variance=mx.nd.ones(shape=M) * 0.5)
        else:
            self.qa = a_prior

        # variational posterior of ability
        mean = Variable(mx.random.normal(shape=C))
        variance = PositiveTransformation().transform(Variable(mx.random.normal(shape=C)))
        base = Normal.define_variable(mean=mean, variance=variance, shape=M)
        self.qtheta = Logistic(lower=0, upper=1).transform(base)

        # variational posterior of difficulty
        mean = Variable(mx.random.normal(shape=C))
        variance = PositiveTransformation().transform(Variable(mx.random.normal(shape=C)))
        base = Normal.define_variable(mean=mean, variance=variance, shape=M)
        self.qtheta = Logistic(lower=0, upper=1).transform(base)
        self.qdelta = TransformedDistribution(distribution=Normal(loc=tf.Variable(tf.random_normal([M])),
                                                                  scale=tf.nn.softplus(
                                                                      tf.Variable(tf.random_normal([M])))), \
                                              bijector=ds.bijectors.Sigmoid(), sample_shape=[C], name='qdelta')

        self.alpha = (transpose(self.qtheta) / self.qdelta) ** self.qa

        self.beta = ((1. - transpose(self.qtheta)) / (1. - self.qdelta)) ** self.qa

        # observed variable
        self.x = Beta.define_variable(transpose(self.alpha), transpose(self.beta))

    def init_inference(self, data, n_iter=1000, n_print=100):

        # for discrimination a is latent variable
        if isinstance(self.a_prior, Variable):
            self.inference = Hierarchi_klqp(latent_vars={self.a_prior: self.qa}, data={self.x: data}, \
                                            local_vars={self.theta_prior: self.qtheta, self.delta_prior: self.qdelta},
                                            local_data={self.x: data})

        # for discrimination a is constant
        else:
            self.inference = Hierarchi_klqp(latent_vars={self.theta_prior: self.qtheta, self.delta_prior: self.qdelta},
                                            data={self.x: data})

        self.inference.initialize(auto_transform=False, n_iter=n_iter, n_print=n_print)

    def fit(self, local_iter=50):

        tf.global_variables_initializer().run()

        for jj in range(self.inference.n_iter):
            if isinstance(self.a_prior, Variable):
                for _ in range(local_iter):
                    self.inference.update(scope='local')

            info_dict = self.inference.update(scope='global')
            self.inference.print_progress(info_dict)


