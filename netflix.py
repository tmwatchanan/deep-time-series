import argparse
import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import ltc_model as ltc
from ctrnn_model import CTGRU, CTRNN, NODE

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Run on CPU


def convert_to_floats(feature_col, memory):
    for i in range(len(feature_col)):
        if(feature_col[i] == "?" or feature_col[i] == "\n"):
            feature_col[i] = memory[i]
        else:
            feature_col[i] = float(feature_col[i])
            memory[i] = feature_col[i]
    return feature_col, memory


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def cut_in_sequences(x, y, seq_len, inc=1):
    sequences_x = []
    sequences_y = []

    for end in range(seq_len, x.shape[0]+1):
        start = end - seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(np.repeat(y[end-1], seq_len))

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)[:, :, None]


class NetflixData:
    def __init__(self, file, seq_len=32):
        self.file = file
        self.X, self.Y = self.read_netflix_stock_data(file)
        print(self.X.min(), self.X.max())
        print(self.Y.min(), self.Y.max())
        self.preprocess_data()

        print(self.X.shape)
        print(self.Y.shape)
        print(self.X.min(), self.X.max())
        print(self.Y.min(), self.Y.max())

        self.split_data(test_index=3930, training_ratio=0.8)
        print(self.X_train.shape, self.X_valid.shape, self.X_test.shape)

        self.X_train_seq, self.Y_train_seq = cut_in_sequences(self.X_train, self.Y_train, seq_len, inc=seq_len)
        self.X_valid_seq, self.Y_valid_seq = cut_in_sequences(self.X_valid, self.Y_valid, seq_len, inc=seq_len)
        self.X_test_seq, self.Y_test_seq = cut_in_sequences(self.X_test, self.Y_test, seq_len, inc=seq_len)
        print(self.X_train_seq.shape, self.X_valid_seq.shape, self.X_test_seq.shape)
        print(self.Y_train_seq.shape, self.Y_valid_seq.shape, self.Y_test_seq.shape)

    def read_netflix_stock_data(self, file_path):
        df = pd.read_csv('data/netflix/NFLX.csv')
        X = df[['High', 'Open', 'Low', 'Volume']].to_numpy()
        Y = df['Close'].to_numpy()
        Y = Y[:, None]
        # for i in range(all_y.shape[1]):
        #     all_x[:, i] = moving_average(all_x[:, i], 3)
        print('X', X.shape)
        print('Y', Y.shape)
        return X, Y

    def preprocess_data(self):
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X = self.scaler_x.fit_transform(self.X)
        self.Y = self.scaler_y.fit_transform(self.Y)

    def split_data(self, test_index, training_ratio=0.8):
        validation_index = int(test_index * training_ratio)
        self.X_train = self.X[:validation_index]
        self.Y_train = self.Y[:validation_index]
        self.X_valid = self.X[validation_index:test_index]
        self.Y_valid = self.Y[validation_index:test_index]
        self.X_test = self.X[test_index:]
        self.Y_test = self.Y[test_index:]

    def iterate_train(self, batch_size=16):
        total_seqs = self.X_train_seq.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.X_train_seq[:, permutation[start:end]]
            batch_y = self.Y_train_seq[:, permutation[start:end]]
            yield (batch_x, batch_y)


class NetflixModel:
    def __init__(self, model_type, model_size, input_size, learning_rate=0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, input_size])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None, 1])

        self.model_size = model_size
        head = self.x
        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif(model_type.startswith("ltc")):
            # learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head, _ = tf.nn.dynamic_rnn(self.wm, head, dtype=tf.float32, time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size, cell_clip=10)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size, cell_clip=-1)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size, cell_clip=-1, global_feedback=True)
            head, _ = tf.nn.dynamic_rnn(self.fused_cell, head, dtype=tf.float32, time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        # target_y = tf.expand_dims(self.target_y,axis=-1)
        self.y = tf.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.TruncatedNormal())(head)
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(self.target_y-self.y))
        lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, 500, 1e-5, 0.5)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        print("ASDASDASDASD")
        print(self.target_y.shape, self.y.shape)
        self.accuracy = tf.reduce_mean(tf.abs(self.target_y-self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results", "netflix", "{}_{}.csv".format(model_type, model_size))
        if(not os.path.exists("results/netflix")):
            os.makedirs("results/netflix")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file, "w") as f:
                f.write("best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n")

        self.checkpoint_path = os.path.join("tf_sessions", "netflix", "{}".format(model_type))
        if(not os.path.exists("tf_sessions/netflix")):
            os.makedirs("tf_sessions/netflix")

        self.saver = tf.train.Saver()

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, data, epochs, verbose=True, log_period=50, batch_size=16):
        print('batch_size', batch_size)

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0)
        self.save()
        for e in range(epochs):
            if(verbose and e % log_period == 0):
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: data.X_test_seq, self.target_y: data.Y_test_seq})
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: data.X_valid_seq, self.target_y: data.Y_valid_seq})
                # MSE metric -> less is better
                if((valid_loss < best_valid_loss and e > 0) or e == 1):
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses), np.mean(accs),
                        valid_loss, valid_acc,
                        test_loss, test_acc
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x, batch_y in data.iterate_train(batch_size=batch_size):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y})
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e % log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                    e, np.mean(losses), np.mean(accs), valid_loss, valid_acc, test_loss, test_acc))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
            best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))
        test_predicted = self.sess.run(self.y, {self.x: data.X_test_seq})
        print('>>>>>>>>>>>>>>>>>>>')
        print(test_predicted.shape)

        y_true = np.transpose(data.Y_test_seq, (1, 0, 2))
        print('test_predicted', test_predicted.shape)
        y_pred = np.transpose(test_predicted, (1, 0, 2))
        y_true = y_true[:, -1]
        y_pred = y_pred[:, -1]
        print(y_true.shape, y_pred.shape)
        y_true = data.scaler_y.inverse_transform(y_true)
        y_pred = data.scaler_y.inverse_transform(y_pred)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_true, color='black', linewidth=.9)
        ax.plot(y_pred, color='green', linewidth=.8, linestyle='dashed')
        ax.set_xlim(0, len(y_true)-1)
        plt.savefig(f'netflix-{self.model_type}_{self.model_size}.png', dpi=300, bbox_inches='tight')

        with open(self.result_file, "a") as f:
            f.write("{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='data/netflix/NFLX.csv')
    parser.add_argument('--model', default="lstm")
    parser.add_argument('--log', default=1, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    args = parser.parse_args()

    data = NetflixData(args.file)
    model = NetflixModel(model_type=args.model, model_size=args.size,
                         input_size=data.X_train.shape[-1], learning_rate=args.lr)

    model.fit(data, epochs=args.epochs, log_period=args.log, batch_size=args.batch_size)
