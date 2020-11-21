import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd


class MLP:
    def __init__(
            self,
            num_layers,
            neuron_additional_neurons,
            learning_rate=0.01,
            batch_size=100,
            epoch=100,
            verbose=False
    ):
        self.ctx = mx.cpu()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epoch
        self.num_layers = num_layers
        self.neuron_additional_neurons = neuron_additional_neurons
        self.verbose = verbose
        self.net = gluon.nn.HybridSequential()

    def evaluate_accuracy(self, data_iterator):
        rmse = mx.metric.RMSE()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.net(data)
            predictions = output
            rmse.update(preds=predictions, labels=label)
        return rmse.get()[1]

    def _infer_neuron_list(self, num_features, num_output, num_layers, neuron_additional_neurons):
        inferred_first_layer_neurons = int(np.sqrt(num_features + num_output))
        step_length = int(float(inferred_first_layer_neurons - num_output) / num_layers)
        layer_list = list(range(inferred_first_layer_neurons, 1, -step_length))

        layer_list = [layer_neuron + neuron_additional_neurons for layer_neuron in layer_list]
        return layer_list

    def fit(self, x, y):
        x = x.astype('float32')
        y = y.astype('float32')

        layer_list = self._infer_neuron_list(np.shape(x)[1], 1, self.num_layers, self.neuron_additional_neurons)
        with self.net.name_scope():
            for n_neurons in layer_list:
                self.net.add(gluon.nn.Dense(n_neurons, activation="relu"))
                self.net.add(gluon.nn.BatchNorm())
            self.net.add(gluon.nn.Dense(1))

        self.net.hybridize()
        self.net.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx, force_reinit=True)


        l2_loss = gluon.loss.L2Loss()
        trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': self.learning_rate})
        data_set = mx.gluon.data.dataset.ArrayDataset(x, y)
        data_loader = mx.gluon.data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            cumulative_loss = 0
            for i, (data, label) in enumerate(data_loader):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with autograd.record():
                    output = self.net(data)
                    loss = l2_loss(output, label)
                loss.backward()
                trainer.step(data.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()

            train_rmse = self.evaluate_accuracy(data_loader)
            if self.verbose:
                print("Epoch %s, Train_rmse %s" %
                      (epoch, train_rmse))

    def predict(self, x):
        x = x.astype('float32')
        data_set = mx.gluon.data.dataset.ArrayDataset(x)
        data_loader = mx.gluon.data.DataLoader(data_set, batch_size=x.shape[0])
        preds = []
        with autograd.record():
            for data in data_loader:
                preds.append(self.net(data).asnumpy())
        preds = np.concatenate(preds)
        return preds
