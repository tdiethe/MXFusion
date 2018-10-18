from mxfusion.components.distributions import Normal, NormalMeanPrecision
from mxfusion.components.functions.operator_impl import dot
from mxfusion.inference import StochasticVariationalInference
from mxfusion.inference.grad_based_inference import GradBasedInference
from mxfusion.inference import BatchInferenceLoop
from mxfusion.inference.meanfield import create_Gaussian_meanfield
from mxfusion import Model, Posterior, Variable

from IPython.display import display
import numpy as np
import pandas as pd
import mxnet as mx
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_style('darkgrid')


def plot_signals(x, num_to_show):
    num_rows_cols = int(np.sqrt(num_to_show))
    fig, axs = plt.subplots(num_rows_cols, num_rows_cols,
                            figsize=(num_rows_cols * 5, num_rows_cols * 5))
    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].plot(range(n), x[i, :])
        axs[i].set(xticklabels=[])
        axs[i].set(yticklabels=[])
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)


data = loadmat('/Users/tdiethe/code/MXFusion/examples/notebooks/data.mat')

# sns.lineplot(x=range(data['train_x'].shape[1]), y=data['train_x'][0, :])

df_train = pd.DataFrame(np.hstack([data['train_x'], data['train_y'].T]),
                        columns=tuple(map(str, range(data['train_x'].shape[1]))) + ('y',))
# display(df_train.head())
df1_train = df_train[df_train['y'] == 1].drop('y', inplace=False, axis=1)
# display(df1_train.head())

df_test = pd.DataFrame(np.hstack([data['test_x'], data['test_y'].T]),
                       columns=tuple(map(str, range(data['test_x'].shape[1]))) + ('y',))
# display(df_test.head())
df1_test = df_test[df_test['y'] == 1].drop('y', inplace=False, axis=1)
# display(df1_test.head())

x_train = df1_train.values
x_test = df1_test.values

[m_train, n] = x_train.shape
m_test = x_test.shape[0]
print(x_train.shape)
# print(y_train.shape)

# MXFusion model

k = 64

m = Model()
zero = mx.nd.array([0])
one = mx.nd.array([1])
noise_variance = mx.nd.array([0.1])

# m.Z = Normal.define_variable(mean=zero, variance=one, shape=(m_train, k))
m.Z = NormalMeanPrecision.define_variable(mean=zero, variance=one, shape=(m_train, k))
m.D = Normal.define_variable(mean=zero, variance=one, shape=(k, n))
m.X = Normal.define_variable(
    mean=dot(m.Z, m.D, shape=(m_train, n)),
    variance=noise_variance,
    shape=(m_train, n))
# m.X.set_prior(distribution=)


observed = [m.X]
q = create_Gaussian_meanfield(model=m, observed=observed)

vi = StochasticVariationalInference(model=m, num_samples=3, observed=observed, posterior=q)
inf = GradBasedInference(inference_algorithm=vi, grad_loop=BatchInferenceLoop())

inf.params[m.D] = mx.nd.random.normal(loc=0, scale=100, shape=m.D.shape)
inf.params[m.Z] = mx.nd.random.normal(loc=0, scale=100, shape=m.Z.shape)

inf.run(X=mx.nd.array(df1_train.values), max_iter=1000, verbose=True)

D_post_mean = inf.params[q.D.factor.mean].asnumpy()
D_post_var = inf.params[q.D.factor.variance].asnumpy()

plot_signals(D_post_mean, 64)

plt.show()

print()
