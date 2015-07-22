#coding:utf-8
import pylab
import numpy as np
import matplotlib.pyplot as plt
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

    aveLoop = 30
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

        #-------------------------------
        # 訓練データ、テストデータを用意する
        #-------------------------------
        # 全ての数字を含むテスト用のデータを用意
        # 訓練データとテストデータに分解 (使うのはテストデータのみ)
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.1)


        # 4を覗いたインデックスを取得
        index_without4 = np.where((y != 4))
        # 4を除いたインデックスの教師信号を取得
        y_without4 = y[index_without4]
        # 4を除いたインデックスの入力を取得
        X_without4 = X[index_without4]
        # 4を除いた訓練とテストそれぞれのデータを生成
        X_train_without4, X_test_without4, y_train_without4, y_test_without4 = train_test_split(X_without4, y_without4, test_size=0.1)
        # 教師信号の数字を1-of-K表記に変換
        labels_train_without4 = LabelBinarizer().fit_transform(y_train_without4)
        labels_test_without4 = LabelBinarizer().fit_transform(y_test_without4)
        # データの値をひとつ減らす(4を抜いたりする)と、1-of-Kも一桁減ってしまうため
        # 0列を4番目に挿入する
        # 列の挿入 (行列, 挿入したい場所, 挿入したい値, axis=1)
        labels_train_without4 = np.insert(labels_train_without4, 4, 0, axis=1)
        labels_test_without4 = np.insert(labels_test_without4, 4, 0, axis=1)



        # 教師信号が4であるインデックスを取得
        index_4 = np.where((y == 4))
        # 4のみの教師信号を取得
        y_4 = y[index_4]
        # 4の教師信号に対応する入力を取得
        X_4 = X[index_4]
        # 4のみの訓練とテストそれぞれのデータを生成
        X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, y_4, test_size=0.1)
        # 教師信号の数字を1-ofK表記に変換(このままだと、0が1つのみのリストになっている)
        labels_train_4 = LabelBinarizer().fit_transform(y_train_4)
        labels_test_4 = LabelBinarizer().fit_transform(y_test_4)
        # 列の挿入 (行列, 挿入したい場所, 挿入したい値, axis=1)
        # [0]のようなリストになっているため、1と0を挿入して1-of-kデータを作成する.
        for i in range(9):
            if i == 4:
                labels_train_4 = np.insert(labels_train_4, 4, 1, axis=1)
                labels_test_4 = np.insert(labels_test_4, 4, 1, axis=1)
            else:
                labels_train_4 = np.insert(labels_train_4, i, 0, axis=1)
                labels_test_4 = np.insert(labels_test_4, i, 0, axis=1)

        """
        # データのと教師信号のテスト描画
        plt.imshow(X_train_4[6].reshape(28, 28))
        plt.gray()
        plt.show()
        print labels_train_4[32]
        print labels_test_4[5]
        exit()
        """

        trainNum = 3000

        # 先行学習を行う
        mlp.fit(X_train_without4, labels_train_without4, learning_rate=0.01, epochs=trainNum, xtest=X_test_without4, ytest=y_test_without4)

        # 結果の表示
        # テストデータを用いて予測精度を計算
        predictions = []
        for i in range(X_test_all.shape[0]):
            o = mlp.predict(X_test_all[i])
            predictions.append(np.argmax(o))
        print confusion_matrix(y_test_all, predictions)
        print classification_report(y_test_all, predictions)#target_names = target_names)

        # 追加学習を行う
        mlp.fit(X_train_4, labels_train_4, learning_rate=0.01, epochs=trainNum, xtest=X_test_all, ytest=y_test_all)

        # 結果の表示
        # テストデータを用いて予測精度を計算
        predictions = []
        for i in range(X_test_all.shape[0]):
            o = mlp.predict(X_test_all[i])
            predictions.append(np.argmax(o))
        print confusion_matrix(y_test_all, predictions)
        print classification_report(y_test_all, predictions)
