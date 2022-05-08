# -*- coding: utf-8 -*-
"""CTRCVR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10tVYQyAJZH_ZK3mC-0X_fViLOdxG8Hl9
"""

from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import *
from tensorflow.keras import optimizers
import time
import os
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


class CTCVRNet:
    def __init__(self):
        pass

    def build_ctr_model(self, ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                        ctr_item_cate_input, ctr_user_cate_feature_dict, ctr_item_cate_feature_dict):

        user_feature = layers.Dropout(0.5)(ctr_user_numerical_input)
        user_feature = layers.BatchNormalization()(user_feature)
        user_feature = layers.Dense(128, activation='relu')(user_feature)
        user_feature = layers.Dense(64, activation='relu')(user_feature)

        item_feature = layers.Dropout(0.5)(ctr_item_numerical_input)
        item_feature = layers.BatchNormalization()(item_feature)
        item_feature = layers.Dense(128, activation='relu')(item_feature)
        item_feature = layers.Dense(64, activation='relu')(item_feature)

        dense_feature = layers.concatenate(
            [user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(0.5)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(64, activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid',
                            name='ctr_output')(dense_feature)
        return pred

    def build_cvr_model(self, cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                        cvr_item_cate_input, cvr_user_cate_feature_dict, cvr_item_cate_feature_dict):

        user_feature = layers.Dropout(0.5)(cvr_user_numerical_input)
        user_feature = layers.BatchNormalization()(user_feature)
        user_feature = layers.Dense(128, activation='relu')(user_feature)
        user_feature = layers.Dense(64, activation='relu')(user_feature)

        item_feature = layers.Dropout(0.5)(cvr_item_numerical_input)
        item_feature = layers.BatchNormalization()(item_feature)
        item_feature = layers.Dense(128, activation='relu')(item_feature)
        item_feature = layers.Dense(64, activation='relu')(item_feature)

        dense_feature = layers.concatenate(
            [user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(0.5)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(64, activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid',
                            name='cvr_output')(dense_feature)
        return pred

    def build(self, user_cate_feature_dict, item_cate_feature_dict):
        # CTR model input
        ctr_user_numerical_input = layers.Input(shape=(user_feature_num,))
        ctr_user_cate_input = layers.Input(shape=(0,))
        ctr_item_numerical_input = layers.Input(shape=(item_feature_num,))
        ctr_item_cate_input = layers.Input(shape=(0,))

        # CVR model input
        cvr_user_numerical_input = layers.Input(shape=(user_feature_num,))
        cvr_user_cate_input = layers.Input(shape=(0,))
        cvr_item_numerical_input = layers.Input(shape=(item_feature_num,))
        cvr_item_cate_input = layers.Input(shape=(0,))

        ctr_pred = self.build_ctr_model(ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                                        ctr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
        cvr_pred = self.build_cvr_model(cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                                        cvr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
        ctcvr_pred = tf.multiply(ctr_pred, cvr_pred)
        model = Model(
            inputs=[ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input],
            outputs=[ctr_pred, ctcvr_pred])

        return model

    def preprocess_data(self, data, labels):

        sample_num = data.shape[0]
        if sample_num % 2 == 1:
            sample_num -= 1
            data = data[1:]

        half_data_num = int(sample_num/2)
        train_sample_num = int(half_data_num*0.6)
        val_sample_num = int(half_data_num*0.2)
        test_sample_num = half_data_num - train_sample_num - val_sample_num
        feature_num = data.shape[1]-len(labels)

        features = list(set(data.columns)-set(labels))
        user_feature_num = feature_num//2
        item_feature_num = feature_num-user_feature_num

        user_features = features[:user_feature_num]
        item_features = features[user_feature_num:]

        data0 = data[:half_data_num]
        data1 = data[half_data_num:]

        X0_data_user = data0[user_features]
        X1_data_user = data1[user_features]
        X0_data_item = data0[item_features]
        X1_data_item = data1[item_features]
        Y0_data = data0[labels[0]]
        Y1_data = data1[labels[1]]

        X0_data_user_test = X0_data_user[train_sample_num+val_sample_num:]
        X1_data_user_test = X1_data_user[train_sample_num+val_sample_num:]
        X0_data_item_test = X0_data_item[train_sample_num+val_sample_num:]
        X1_data_item_test = X1_data_item[train_sample_num+val_sample_num:]
        Y0_data_test = Y0_data[train_sample_num+val_sample_num:]
        Y1_data_test = Y1_data[train_sample_num+val_sample_num:]

        X0_data_user_val = X0_data_user[train_sample_num:train_sample_num+val_sample_num]
        X1_data_user_val = X1_data_user[train_sample_num:train_sample_num+val_sample_num]
        X0_data_item_val = X0_data_item[train_sample_num:train_sample_num+val_sample_num]
        X1_data_item_val = X1_data_item[train_sample_num:train_sample_num+val_sample_num]
        Y0_data_val = Y0_data[train_sample_num:train_sample_num+val_sample_num]
        Y1_data_val = Y1_data[train_sample_num:train_sample_num+val_sample_num]

        X0_data_user = X0_data_user[:train_sample_num]
        X1_data_user = X1_data_user[:train_sample_num]
        X0_data_item = X0_data_item[:train_sample_num]
        X1_data_item = X1_data_item[:train_sample_num]
        Y0_data = Y0_data[:train_sample_num]
        Y1_data = Y1_data[:train_sample_num]

        train_data = [X0_data_user, pd.DataFrame(np.zeros(train_sample_num)), X0_data_item,
                      pd.DataFrame(np.zeros(train_sample_num)), X1_data_user, pd.DataFrame(
                          np.zeros(train_sample_num)),
                      X1_data_item, pd.DataFrame(np.zeros(train_sample_num)), pd.DataFrame(Y0_data), pd.DataFrame(Y1_data)]

        val_data = [X0_data_user_val, pd.DataFrame(np.zeros(val_sample_num)), X0_data_item_val,
                    pd.DataFrame(np.zeros(val_sample_num)), X1_data_user_val, pd.DataFrame(
                        np.zeros(val_sample_num)),
                    X1_data_item_val, pd.DataFrame(np.zeros(val_sample_num)), pd.DataFrame(Y0_data_val), pd.DataFrame(Y1_data_val)]

        test_data = [X0_data_user_test, pd.DataFrame(np.zeros(test_sample_num)), X0_data_item_test,
                     pd.DataFrame(np.zeros(test_sample_num)), X1_data_user_test, pd.DataFrame(
                         np.zeros(test_sample_num)),
                     X1_data_item_test, pd.DataFrame(np.zeros(test_sample_num)), pd.DataFrame(Y0_data_test), pd.DataFrame(Y1_data_test)]
        return train_data, val_data, test_data

    def train(self, data, labels, plot_list=[], verbose=0, epoches=20, batchsize=128):
        """
        model train and save as tf serving model
        :param cate_feature_dict: dict, categorical feature for data
        :param user_cate_feature_dict: dict, user categorical feature
        :param item_cate_feature_dict: dict, item categorical feature
        :param train_data: DataFrame, training data
        :param val_data: DataFrame, valdation data
        :return: None
        """
        cate_feature_dict = {}
        user_cate_feature_dict = {}
        item_cate_feature_dict = {}
        train_data, val_data, test_data = self.preprocess_data(data, labels)

        ctcvr = CTCVRNet()
        ctcvr_model = ctcvr.build(
            user_cate_feature_dict, item_cate_feature_dict)
        opt = optimizers.Adam(lr=0.003, decay=0.0001)
        ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                            metrics=[tf.keras.metrics.AUC()])

        # keras model save path
        filepath = "esmm_best.h5"

        # call back function
        checkpoint = ModelCheckpoint(
            filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=0)
        earlystopping = EarlyStopping(
            monitor='val_loss', min_delta=0.0001, patience=8, verbose=0, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]

        # load data
        ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train, \
            ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train, \
            cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train = train_data

        ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val, \
            ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val, \
            cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val = val_data

        ctr_user_numerical_feature_test, ctr_user_cate_feature_test, ctr_item_numerical_feature_test, \
            ctr_item_cate_feature_test, cvr_user_numerical_feature_test, cvr_user_cate_feature_test, \
            cvr_item_numerical_feature_test, cvr_item_cate_feature_test, ctr_target_test, cvr_target_test = test_data

        # model train
        history = ctcvr_model.fit([ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
                                  ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                                  cvr_item_numerical_feature_train,
                                  cvr_item_cate_feature_train], [ctr_target_train, cvr_target_train], batch_size=batchsize, epochs=epoch,
                                  validation_data=(
            [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
             ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
             cvr_item_numerical_feature_val,
             cvr_item_cate_feature_val], [ctr_target_val, cvr_target_val]),
            #  callbacks=callbacks,
            verbose=1,
            shuffle=True)

        plt.figure()
        plt.plot(history.history['loss'])
        plt.title('Loss curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

        predictions = ctcvr_model.predict([ctr_user_numerical_feature_test, ctr_user_cate_feature_test, ctr_item_numerical_feature_test,
                                          ctr_item_cate_feature_test, cvr_user_numerical_feature_test, cvr_user_cate_feature_test,
                                          cvr_item_numerical_feature_test, cvr_item_cate_feature_test])

        FPR, TPR, threshold = roc_curve(
            ctr_target_test, predictions[0].reshape(-1))

        AUC = auc(FPR, TPR)

        plt.figure()
        plt.title('Y1 ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.show()

        plt.figure()
        precision, recall, thresholds = precision_recall_curve(
            ctr_target_test, predictions[0].reshape(-1))
        plt.title('Y1 PR CURVE')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(recall, precision)
        plt.show()

        FPR, TPR, threshold = roc_curve(
            cvr_target_test, predictions[1].reshape(-1))

        AUC = auc(FPR, TPR)

        plt.figure()
        plt.title('Y2 ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(FPR, TPR, color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.show()

        plt.figure()
        precision, recall, thresholds = precision_recall_curve(
            cvr_target_test, predictions[1].reshape(-1))
        plt.title('Y2 PR CURVE')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(recall, precision)
        plt.show()
        # load model and save as tf_serving model
        # saved_model_path = './esmm/{}'.format(int(time.time()))
        # ctcvr_model = tf.keras.models.load_model('esmm_best.h5')
        # tf.saved_model.save(ctcvr_model, saved_model_path)
