#coding:utf-8
import numpy as np
from mlp import MultiLayerPerceptron
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import csv
import os

"""
MNISTの手書き数字データの認識
scikit-learnのインストールが必要
http://scikit-learn.org/
"""

if __name__ == "__main__":

    aveLoop = 1
    average = np.array([0]*16) #平均計算用8+8

    # MNISTの数字データ
    # 70000サンプル, 28x28ピクセル
    # カレントディレクトリ（.）にmnistデータがない場合は

    # data_homeに存在しない場合, Webから自動的にダウンロードされる（時間がかかる）
    mnist = fetch_mldata('MNIST original', data_home="./../../../")

    # 訓練データを作成
    X = mnist.data
    y = mnist.target

    # ピクセルの値を0.0-1.0に正規化
    X = X.astype(np.float64)
    X /= X.max()

    # 保存ファイルがない場合、作成する.
    if not os.path.isdir("./precision"):
        os.mkdir("./precision")
    if not os.path.isdir("./recall"):
        os.mkdir("./recall")
    if not os.path.isdir("./F-score"):
        os.mkdir("./F-score")


    # 平均の回数だけ繰り返す
    for loop in range(aveLoop):
        print "***",loop,"***"

        # ファイルの作成
        # precision
        preMat = open("./precision/preMat"+str(loop)+".csv", "w")
        preMat = csv.writer(preMat)
        # recall
        recMat = open("./recall/recMat"+str(loop)+".csv", "w")
        recMat = csv.writer(recMat)
        # fscore
        fscMat = open("./F-score/fscMat"+str(loop)+".csv", "w")
        fscMat = csv.writer(fscMat)

        # 多層パーセプトロンを構築
        mlp = MultiLayerPerceptron(28*28, 1000, 10, act1="tanh", act2="softmax", preMat=preMat, recMat=recMat, fscMat=fscMat)

        #####################
        # 0~9の全てのデータ
        #####################

        # 訓練データとテストデータに分解
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # 教師信号の数字を1-of-K表記に変換(全てのデータ)
        labels_train = LabelBinarizer().fit_transform(y_train)
        labels_test = LabelBinarizer().fit_transform(y_test)


        # 教師信号の数字を1-of-K表記に変換
        labels_train = LabelBinarizer().fit_transform(y_train)
        labels_test = LabelBinarizer().fit_transform(y_test)

        trainNum = 5000

        # 訓練データを用いてニューラルネットの重みを学習
        mlp.fit(X_train, labels_train, learning_rate=0.01, epochs=trainNum, xtest=X_test, ytest=y_test)

        # 結果の表示
        # テストデータを用いて予測精度を計算
        predictions = []
        for i in range(X_test.shape[0]):
            o = mlp.predict(X_test[i])
            predictions.append(np.argmax(o))
