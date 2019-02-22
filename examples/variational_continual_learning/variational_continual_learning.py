# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

import numpy as np
import mxnet as mx

import matplotlib.pyplot as plt
from datetime import datetime

from examples.variational_continual_learning.experiment import Experiment
from examples.variational_continual_learning.mnist import SplitTaskGenerator, PermutedTaskGenerator
from examples.variational_continual_learning.coresets import Random, KCenter, Vanilla

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

# Set the compute context, GPU is available otherwise CPU
CTX = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


def set_seeds(seed=42):
    mx.random.seed(seed)
    np.random.seed(seed)


def plot(title, experiments, num_tasks):
    fig = plt.figure(figsize=(num_tasks, 3))
    ax = plt.gca()

    x = range(1, len(tasks) + 1)

    for experiment in experiments:
        acc = np.nanmean(experiment.overall_accuracy, axis=1)
        label = experiment.coreset.__class__.__name__
        plt.plot(x, acc, label=label, marker='o')
    ax.set_xticks(x)
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('# tasks')
    ax.legend()
    ax.set_title(title)

    filename = "vcl_{}_{}.pdf".format(title, datetime.now().isoformat()[:-7])
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load data
    data = mx.test_utils.get_mnist()
    input_dim = int(np.prod(data['train_data'][0].shape))  # Note the data will get flattened later
    verbose = False

    # noinspection PyUnreachableCode
    if True:
        title = "Split MNIST"
        tasks = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        num_epochs = 120
        # tasks = ((0, 1), (2, 3))
        # num_epochs = 1  # 120
        batch_size = None
        gen = SplitTaskGenerator
        label_shape = 2
        network_shape = (input_dim, 256, 256, (label_shape, ))
        single_head = False
        coreset_size = 40
    else:
        title = "Permuted MNIST"
        tasks = range(10)
        num_epochs = 100
        # tasks = range(2)
        # num_epochs = 1
        batch_size = 256
        gen = PermutedTaskGenerator
        label_shape = 10
        network_shape = (input_dim, 100, 100, label_shape)
        single_head = True
        coreset_size = 200

    data_dtype = data['train_data'].dtype
    label_dtype = data['train_label'].dtype

    learning_rate = 0.01
    optimizer = 'adam'

    experiment_parameters = (
        dict(
            coreset=Vanilla(),
            learning_rate=learning_rate,
            optimizer=optimizer,
            network_shape=network_shape,
            num_epochs=num_epochs,
            single_head=single_head),
        dict(
            coreset=Random(coreset_size=coreset_size),
            learning_rate=learning_rate,
            optimizer=optimizer,
            network_shape=network_shape,
            num_epochs=num_epochs,
            single_head=single_head),
        dict(
            coreset=KCenter(coreset_size=coreset_size),
            learning_rate=learning_rate,
            optimizer=optimizer,
            network_shape=network_shape,
            num_epochs=num_epochs,
            single_head=single_head)
    )

    experiments = []

    # Run experiments
    for params in experiment_parameters:
        print("-" * 50)
        print("Running experiment", params['coreset'].__class__.__name__)
        print("-" * 50)
        set_seeds()
        experiment = Experiment(batch_size=batch_size,
                                data_generator=gen(data, batch_size=batch_size, tasks=tasks),
                                ctx=CTX, verbose=verbose,
                                **params)
        experiment.run()
        print(experiment.overall_accuracy)
        experiments.append(experiment)
        print("-" * 50)
        print()

    plot(title, experiments, len(tasks))