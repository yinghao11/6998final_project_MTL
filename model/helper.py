# -*- coding: utf-8 -*-
"""helper.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1h1tga-4yyVWzaiihn1IN93Gmm1YlKMzj
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
from sklearn.metrics import roc_curve,auc,precision_recall_curve,roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import collections


import matplotlib.pyplot as plt
from mmoe import MMoE

def custom_loss(y_true, y_pred):
    y_true=tf.cast(y_true,tf.float32)
    not_ignore_case=(y_true[:,0]!=-1)

    y_true=y_true[not_ignore_case]
    y_pred=y_pred[not_ignore_case]
    diff = K.square(y_pred- y_true)  #squared difference
    loss = K.mean(diff, axis=-1) #mean over last dimension
    return loss

def filter_label(df,labels, default_value=-1):
    for label in labels:
        if label not in df.columns:
            df[label]=default_value
    return df
    
def get_data(folderpath, labels):
    # build synthetic training data
    dataset_filenemes=os.listdir(folderpath)
    dataset_filenemes=[f for f in dataset_filenemes if os.path.isfile(os.path.join(folderpath,f))]
    print(f"Datasets used : {list(dataset_filenemes)}")
    
    data=pd.read_csv(os.path.join(folderpath,dataset_filenemes[0]))
    data=filter_label(data, labels)
    data=data.fillna(0)
    all_features=list(set(data.columns)-set(labels))
    print(f"All features we used: {all_features if len(all_features)<=10 else all_features[:10]} (at most 10)")
    for i,file_name in enumerate(dataset_filenemes[1:]):
        _data=pd.read_csv(os.path.join(folderpath,file_name))
        _data=_data.fillna(0)
        _data=filter_label(_data, all_features,default_value=0)
        _data=filter_label(_data, labels)
        _data=_data[all_features+labels]

        data=data.append(_data)

    data=data.sample(frac=1)
    return data

def cus_accuracy(y_true, y_pred):
    y_true=tf.cast(y_true,tf.float32)
    not_ignore_case=(y_true[:,0]!=-1)

    y_true=y_true[not_ignore_case]
    y_pred=y_pred[not_ignore_case]
    
    y_pred=K.round(K.clip(y_pred,0,1))
    return K.cast(K.equal(y_true, y_pred), K.floatx())