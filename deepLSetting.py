#!/usr/bin/env python
# coding: utf-8

# In[30]:
import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras.optimizers as opt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pyswarms as ps
from pyswarms.utils.search import GridSearch
import horovod.keras as hvd
#機械学習のハイパーパラメータ、学習方法などをまとめたクラス。
class DeepLSetting:
    
    def __init__(self):
        self.num_featureValue = None #特徴量の数
        self.num_output = None #目的変数の数
        self.trainRow = None #学習データの割合
        self.model = Sequential()#学習モデル。
        self.num_nodeList = None #レイヤーの配列。
        #self.strategy = tf.distribute.MirroredStrategy() #分散ストラテジーの作成
    
    #説明変数の数、目的変数の数、学習データの割合を設定する。
    def set_initial(self,  num_featureValue, num_output, trainRow):
        self.num_featureValue = num_featureValue
        self.num_output = num_output
        self.trainRow = trainRow
    
        
    #modelにNN構造を手動で設定する
    def set_modelLayerAndNode(self,num_nodeList: list, activation="relu", dropout=0.2):
        self.num_nodeList = num_nodeList.copy()
        self.model.add(Dense(num_nodeList[1], input_shape=(num_nodeList[0],)))
        #self.model.add(Activation(activation))
        self.model.add(Dropout(dropout))
        if(len(num_nodeList) == 2):
            return
        for i in range(2, len(num_nodeList)):
            self.model.add(Dense(num_nodeList[i]))
            #self.model.add(Activation(activation))
            if i < len(num_nodeList)-1:
                self.model.add(Dropout(dropout))

    #モデルをコンパイルする。
    def model_compile(self, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam(), metrics_tmp=["mae"],run_eagerly=True):
        optimizer_tmp = hvd.DistributedOptimizer(optimizer_tmp)
        self.model.compile(
        loss = loss_tmp,
        optimizer=optimizer_tmp,
        metrics=metrics_tmp,
        experimental_run_tf_function=False)

    #ベイズ最適化用の関数
    def func(self, num_layer, num_node, dropout, batch, data=None, num_epoch=3000, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam(), k_fold=0):
        data_tmp = data.copy()
        nodeList = [self.num_featureValue]
        for i in range(1, int(num_layer)):
            if(i == int(num_layer)-1):
                nodeList.append(self.num_output)
                break
            nodeList.append(int(num_node))
        with self.strategy.scope():
            self.model = Sequential() #モデルの初期化
            self.set_modelLayerAndNode(nodeList, dropout=dropout) #モデル構造の定義
            self.model_compile(loss_tmp=loss_tmp, optimizer_tmp=optimizer_tmp)
        #データの準備
        X = data_tmp.iloc[:, 0:self.num_featureValue]
        y = data_tmp.iloc[:, self.num_featureValue:len(data_tmp.columns)]
        X_train = X[self.trainRow[0]:self.trainRow[1]]
        X_test = X[self.trainRow[1]:len(X)]
        y_train = y[self.trainRow[0]:self.trainRow[1]]
        y_test = y[self.trainRow[1]:len(X)]
        #K分割交差検証
        if k_fold:
            return -40000*self.k_foldCrossValidation(k_fold, X_train, y_train, nodeList, dropout, num_epoch=num_epoch, batch=int(batch), loss_tmp=loss_tmp, optimizer_tmp=optimizer_tmp)
        #学習
        with self.strategy.scope():
            history = self.model.fit(X_train, y_train, batch_size=int(batch), epochs=num_epoch, verbose=0)
            score = self.model.evaluate(X_test, y_test, verbose=0)
        return -40000*score[0]
    
    def func_ps(self, x, data=None, num_epoch=3000, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam()):
        y = []
        for i in range(len(x)):
            score = -1*self.func(x[i,0], x[i,1], x[i,2], x[i, 3], data=data, num_epoch=num_epoch, loss_tmp=loss_tmp, optimizer_tmp=optimizer_tmp)
            y.append(score)
        return y

    #NN構造をレイヤー数、ノード数、ドロップアウト、バッチ数を最適化する。(ベイズ最適化)
 
    def bayesOpt(self, data, pbounds,num_epoch=3000, n_iter=25, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam(), k_fold=0):
        if(self.num_featureValue == None):
            raise Exception("先にset_initial関数で初期化してください。")
        self.func.__func__.__defaults__ = data, num_epoch, loss_tmp, optimizer_tmp, k_fold
        optimizer = BayesianOptimization(f=self.func, pbounds=pbounds)
        optimizer.maximize(init_points=5, n_iter=n_iter)
    #NN構造をレイヤー数、ノード数、ドロップアウト、バッチ数を最適化する。(PSO最適化)
    def psoOpt(self, data,num_epoch=3000, n_iter=25, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam()):
        if(self.num_featureValue == None):
            raise Exception("先にset_initial関数で初期化してください。")
        self.func_ps.__func__.__defaults__ = data, num_epoch, loss_tmp, optimizer_tmp
        # Grid search による最適化関数各項の重みの最適化
        #options = {"c1": [0.3, 0.5, 0.8], "c2": [0.3, 0.5, 0.8], "w": [0.2, 0.3, 0.5]}
        #g_search = GridSearch(ps.single.GlobalBestPSO, n_particles = 10, dimensions = 4,
        #options = options, objective_func = self.func_ps, iters = 5)
        #best_score, best_options = g_search.search()
        #print(f"best_options: {best_options}")
        data = data.copy()
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2} #パラメータを設定
        bounds=((3, 2, 0.1, 1), (10, 4096, 0.4, 5)) #範囲を指定
        optimizer = ps.single.LocalBestPSO(n_particles=5, dimensions=4, options=options, bounds=bounds)
        cost, pos = optimizer.optimize(self.func_ps, n_iter, verbose=1)
        f = open('sample.txt', 'w', encoding='UTF-8')
        f.write(f"cost:{cost}")
        f.write(f"pos:{pos}")
        f.close()

    #K分割交差検証
    def k_foldCrossValidation(self, k, X_train, y_train, nodeList, dropout, num_epoch, batch, loss_tmp='mean_squared_error', optimizer_tmp=opt.Adam()):
        kf = KFold(n_splits=k, shuffle=True)
        all_test_loss=[]
        self.model = Sequential() #モデルの初期化
        self.set_modelLayerAndNode(nodeList, dropout=dropout)
        self.model_compile(loss_tmp=loss_tmp, optimizer_tmp=optimizer_tmp)
        for train_index, val_index in kf.split(X_train,y_train):
            train_data=X_train.iloc[train_index,:]
            train_label=y_train.iloc[train_index,:]
            test_data=X_train.iloc[val_index,:]
            test_label=y_train.iloc[val_index,:]
            history=self.model.fit(train_data,
                            train_label,
                            epochs=num_epoch,
                            batch_size=batch,
                            verbose=0)
            score = self.model.evaluate(test_data, test_label, verbose=0)
            all_test_loss.append(score[0])
        ave_all_test_loss = np.mean(all_test_loss)
        return ave_all_test_loss


