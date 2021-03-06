# -*- coding: utf-8 -*-
"""mmoe.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tf8f4YF4ay7yxWfq9S3XWdn_ZqSm9u8c
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import collections

from time import time
import matplotlib.pyplot as plt
from mmoe import MMoE
from helper import custom_loss, filter_label, get_data, cus_accuracy


class mmoe_model():
    def __init__(self, num_features):
        self.num_features = num_features

        # num_features = train_X.shape[1]

        # Set up the input layer
        input_layer = Input(shape=(num_features,))

        # Set up MMoE layer
        mmoe_layers = MMoE(
            units=16,
            num_experts=8,
            num_tasks=2
        )(input_layer)

        output_layers = []

        output_info = ['y0', 'y1']

        # Build tower layer from MMoE layer
        for index, task_layer in enumerate(mmoe_layers):
            tower_layer = Dense(
                units=8,
                activation='relu',
                kernel_initializer=VarianceScaling())(task_layer)
            output_layer = Dense(
                units=1,
                name=output_info[index],
                activation='linear',
                kernel_initializer=VarianceScaling())(tower_layer)
            output_layers.append(output_layer)

        # Compile model
        model = Model(inputs=[input_layer], outputs=output_layers)
        learning_rates = [1e-4, 1e-3, 1e-2]
        adam_optimizer = Adam(lr=learning_rates[0])
        model.compile(
            # loss={'y0': custom_loss, 'y1': custom_loss},
            loss={'y0': 'mean_squared_error', 'y1': 'mean_squared_error'},
            # loss={'y0': 'binary_crossentropy', 'y1': 'binary_crossentropy'},

            optimizer=adam_optimizer,
            metrics=[metrics.mae, cus_accuracy, "accuracy",   tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy", dtype=None, threshold=0.5)]
        )

        # Print out model architecture summary
        model.summary()

        self.model = model
        print("init done")

    def data_preprocess(self, data, labels):
        data_X = np.array(data.drop(labels, axis=1))
        data_y = np.array(data[labels]).reshape(-1, len(labels))
        dev_data, test_data, dev_label, test_label = train_test_split(
            data_X, data_y, test_size=0.2)
        train_data, validation_data, train_label, validation_label = train_test_split(
            dev_data, dev_label, test_size=0.2)

        train_label = [train_label[:, i] for i in range(train_label.shape[1])]

        validation_label = [validation_label[:, i]
                            for i in range(validation_label.shape[1])]

        test_label = [test_label[:, i] for i in range(test_label.shape[1])]

        return (train_data, train_label), (validation_data, validation_label), (test_data, test_label)

    def train(self, data, labels, plot_list=[], epoches=20, verbose=0):
        train_data, validation_data, test_data = self.data_preprocess(
            data, labels)

        train_X, train_y = train_data
        validation_X, validation_y = validation_data
        test_X, test_y = test_data

        print('Training data shape = {}'.format(train_X.shape))
        print('Validation data shape = {}'.format(validation_X.shape))
        print('Test data shape = {}'.format(test_X.shape))

        start_time = time()
        # Train the model
        history = self.model.fit(
            x=train_X,
            y=train_y,
            validation_data=(validation_X, validation_y),
            epochs=epoches,
            verbose=verbose,
            batch_size=2024
        )
        end_time = time()

        if plot_list:
            self.plot_result(history, plot_list, test_X, test_y)

        print(f"\nTime used to train the model: {end_time-start_time: .5f}s")
        # if val_data:
        #     val_X,val_y=val_data

    def plot_result(self, history, plot_list, test_X, test_y):
        plot_num = len(plot_list)
        plt.subplots(plot_num, figsize=(10, 6*plot_num))

        pid = 1
        plt.subplot(plot_num, 1, pid)
        plt.plot(history.history['y0_loss'],
                 color='blue', label='y0_train_loss')
        plt.plot(history.history['y1_loss'],
                 color='green', label='y1_train_loss')
        plt.plot(history.history['val_y0_loss'],
                 color='orange', label='y0_val_loss')
        plt.plot(history.history['val_y1_loss'],
                 color='red', label='y1_val_loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("MMoe Train result - Loss(mse)")
        plt.legend()

        if "accuracy" in plot_list:
            pid += 1
            plt.subplot(plot_num, 1, pid)
            plt.plot(history.history['y0_cus_accuracy'],
                     color='blue', label='y0_train_accuracy')
            plt.plot(history.history['y1_cus_accuracy'],
                     color='green', label='y1_train_accuracy')

            plt.plot(history.history['val_y0_cus_accuracy'],
                     color='orange', label='y0_val_accuracy')
            plt.plot(history.history['val_y1_cus_accuracy'],
                     color='red', label='y1_val_accuracy')

            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("MMoe Train result - Accuracy")
            plt.legend()

        if "auc" in plot_list:
            pid += 1
            plt.subplot(plot_num, 1, pid)

            y_pred = self.model.predict(test_X)
            FPR1, TPR1, threshold = roc_curve(
                test_y[0].reshape(-1), y_pred[0].reshape(-1), pos_label=1)
            FPR2, TPR2, threshold = roc_curve(
                test_y[1].reshape(-1), y_pred[1].reshape(-1), pos_label=1)
            AUC1 = auc(FPR1, TPR1)
            AUC2 = auc(FPR2, TPR2)

            plt.title(
                f'ROC CURVE (AUC= class 1:{AUC1:.2f}, class 2: {AUC2:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.plot(FPR1, TPR1, label="class 1")
            plt.plot(FPR2, TPR2, label="class 2")
            plt.plot([0, 1], [0, 1], color='m', linestyle='--')
            plt.legend()

        if "pr" in plot_list:
            pid += 1
            plt.subplot(plot_num, 1, pid)
            plt.title('PR curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])

            y_pred = self.model.predict(test_X)

            for i in range(len(y_pred)):
                y_cur_pred = y_pred[i]
                y_cur_test = test_y[i].reshape(-1)

                y_cur_pred = y_cur_pred[y_cur_test != -1]
                y_cur_test = y_cur_test[y_cur_test != -1]

                print(collections.Counter(list(y_cur_test.reshape(-1))))

                precision, recall, thresholds = precision_recall_curve(
                    y_cur_test, y_cur_pred.reshape(-1))
                plt.plot(recall, precision, label=f"class {i+1}")

            plt.legend()

        plt.show()
