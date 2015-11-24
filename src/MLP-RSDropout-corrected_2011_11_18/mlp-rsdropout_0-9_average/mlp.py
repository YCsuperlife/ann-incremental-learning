#coding: utf-8
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

"""
mlp.py
多層パーセプトロン
forループの代わりに行列演算にした高速化版
入力層 - 隠れ層 - 出力層の3層構造で固定（PRMLではこれを2層と呼んでいる）
隠れ層の活性化関数にはtanh関数またはsigmoid logistic関数が使える
出力層の活性化関数にはtanh関数、sigmoid logistic関数、恒等関数、softmax関数が使える
"""

def tanh(x):
    return np.tanh(x)

# このスクリプトではxにtanhを通した値を与えることを仮定
def tanh_deriv(x):
    return 1.0 - x ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# このスクリプトではxにsigmoidを通した値を与えることを仮定
def sigmoid_deriv(x):
    return x * (1 - x)

def identity(x):
    return x

def identity_deriv(x):
    return 1

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp)

class MultiLayerPerceptron:
    def __init__(self, numInput, numHidden, numOutput, act1="tanh", act2="sigmoid", preMat=None, recMat=None, fscMat=None):
        """多層パーセプトロンを初期化
        numInput    入力層のユニット数（バイアスユニットは除く）
        numHidden   隠れ層のユニット数（バイアスユニットは除く）
        numOutput   出力層のユニット数
        act1        隠れ層の活性化関数（tanh or sigmoid）
        act2        出力層の活性化関数（tanh or sigmoid or identity or softmax）
        """

        self.NormalizeConstant = 1.0
        self.preMat = preMat
        self.recMat = recMat
        self.fscMat = fscMat

        #出力ファイルのクラスを記述しておく
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.preMat.writerow(classes)
        self.recMat.writerow(classes)
        self.fscMat.writerow(classes)


        # 引数の指定に合わせて隠れ層の活性化関数とその微分関数を設定
        if act1 == "tanh":
            self.act1 = tanh
            self.act1_deriv = tanh_deriv
        elif act1 == "sigmoid":
            self.act1 = sigmoid
            self.act1_deriv = sigmoid_deriv
        else:
            print "ERROR: act1 is tanh or sigmoid"
            sys.exit()

        # 引数の指定に合わせて出力層の活性化関数とその微分関数を設定
        # 交差エントロピー誤差関数を使うので出力層の活性化関数の微分は不要
        if act2 == "tanh":
            self.act2 = tanh
        elif act2 == "sigmoid":
            self.act2 = sigmoid
        elif act2 == "softmax":
            self.act2 = softmax
        elif act2 == "identity":
            self.act2 = identity
        else:
            print "ERROR: act2 is tanh or sigmoid or softmax or identity"
            sys.exit()

        # バイアスユニットがあるので入力層と隠れ層は+1
        self.numInput = numInput + 1
        self.numHidden =numHidden + 1
        self.numOutput = numOutput

        # 重みを (-1.0, 1.0)の一様乱数で初期化
        self.weight1 = np.random.uniform(-1.0, 1.0, (self.numHidden, self.numInput))  # 入力層-隠れ層間
        self.weight2 = np.random.uniform(-1.0, 1.0, (self.numOutput, self.numHidden)) # 隠れ層-出力層間

        # 中間層の使用回数を初期化(バイアス項込み)
        self.hiddenCount = np.array([0.001]*self.numHidden)

        # 中間層のニューロンの価値を初期化(バイアス項なし)
        #self.hiddenValue = np.random.uniform(-1.0, 1.0, (numHidden))
        self.hiddenValue = np.random.uniform(-1.0, 0.0, (numHidden))

        # 上位x%のユニット＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿を分けるしきい値とRSのリファレンス
        self.threshold = 0.25
        self.temp = 1.0
        #self.ref = 1.0 - self.threshold

        #ニューロンの価値の更新に用いる学習率
        self.alpha = 0.9

    def fit(self, X, t, learning_rate=0.1, epochs=10000, xtest=None, ytest=None):
        self.xtest = xtest
        self.ytest = ytest

        """訓練データを用いてネットワークの重みを更新する"""
        # 入力データの最初の列にバイアスユニットの入力1を追加
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        t = np.array(t)

        getTimes=[5, 10, 50, 100, 500, 1000, 1500, 3000, 5000, 7500, 10000, 15000, 30000, 35000, 50000]
        gt = 0

        # 逐次学習
        # 訓練データからランダムサンプリングして重みを更新をepochs回繰り返す
        for k in range(epochs):

            if k == 5:
                #exit()
                pass
            if k%100 == 0:
                print k

            # 訓練データからランダムに選択する
            i = np.random.randint(X.shape[0])

            x = X[i]

            #maskベクトルを生成
            m = self.makeMask()

            # 入力を順伝播させて中間層の出力を計算
            z = self.act1(np.dot(self.weight1, x)) * m

            # 中間層の出力を順伝播させて出力層の出力を計算
            y = self.act2(np.dot(self.weight2, z))

            # 出力層の誤差を計算（交差エントロピー誤差関数を使用）
            delta2 = y - t[i]

            # 出力と信号のユークリッド距離を求める
            e = np.sqrt(np.power(y-t[i], 2).sum())

            #ニューロンの価値を更新する
            #self.hiddenValue += self.alpha * (-e * """np.abs(z[1:])""" z[1:] * - self.hiddenValue) * m[1:]
            #self.hiddenValue += self.alpha * (-e * np.atleast_2d(self.weight2) - self.hiddenValue) * m[1:]

            #print z

            #print np.atleast_2d(self.weight1).T * delta2
            #exit()
            #ニューロンの価値を更新する
            #self.hiddenValue += self.alpha * (-e - self.hiddenValue) * m[1:]

            # 出力層の誤差を逆伝播させて隠れ層の誤差を計算
            delta1 = self.act1_deriv(z) * np.dot(self.weight2.T, delta2) * m

            # 隠れ層の誤差を用いて隠れ層の重みを更新
            # 行列演算になるので2次元ベクトルに変換する必要がある
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)

            #マスクベクトルを2次元ベクトルに変換(行列演算を行うため)
            m2 = np.atleast_2d(m)
            self.weight1 -= learning_rate * np.dot(delta1.T, x) * m2.T

            # 出力層の誤差を用いて出力層の重みを更新
            temp_z = z
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            self.weight2 -= learning_rate * np.dot(delta2.T, z) * m2


            respons = np.dot(delta2.T, z)

            respons2 = np.sum(np.abs(respons), axis=0)
            respons2 = respons2[1:]

            #self.hiddenValue += self.alpha * actRate * ( (-e)  - self.hiddenValue)
            self.hiddenValue += self.alpha * ( respons2 * (-e)  - self.hiddenValue)

            countOf4 = 0
            predictions = []
            if getTimes[gt] == k+1:
                predictions = []
                for i in range(xtest.shape[0]):
                    o = self.predict(xtest[i])
                    predictions.append(np.argmax(o))

                p, r, f, s = precision_recall_fscore_support(ytest, predictions, beta=0.5)


                if len(p) <= 9:
                    p = np.insert(p, 4, 0)
                if len(r) <= 9:
                    r = np.insert(r, 4, 0)
                if len(f) <= 9:
                    f = np.insert(f, 4, 0)

                self.preMat.writerow(p)
                self.recMat.writerow(r)
                self.fscMat.writerow(f)

                gt += 1

    def predict(self, x):
        """テストデータの出力を予測"""
        x = np.array(x)
        # バイアスの1を追加
        x = np.insert(x, 0, 1)
        # 順伝播によりネットワークの出力を計算

        h = (np.atleast_2d(self.hiddenCount))
        #print self.weight2*(h/h[0][0])

        z = self.act1(np.dot(self.weight1 * (h.T/int(h[0][0])), x))
        y = self.act2(np.dot(self.weight2 * (h/int(h[0][0])), z))

        return y

    def makeMask(self):
        # はじめにバイアス項なしで計算を行って, 最後にバイアス項を付け足す

        # ニューロンの価値を降順にソートする
        sortedValue = np.argsort(self.hiddenValue)
        sortedValue = sortedValue[-1::-1]
        # 上位x%を分けるインデックスを求める
        t_idx = int(self.numHidden * self.threshold)
        # 上位x%を分けるような価値refを求める
        ref = (self.hiddenValue[sortedValue[t_idx]] + self.hiddenValue[sortedValue[t_idx+1]]) / 2.0

        # カウントのバイアス項を除く
        cntArray = self.hiddenCount[1:]
        # RSを計算する(行列計算)
        rs = cntArray * (self.hiddenValue - ref)
        #️ RSを昇順にソートした際のインデックスを得る
        valIndex = np.argsort(rs)
        # 降順のインデックスにする
        valIndex = valIndex[-1::-1]
        # 上位50%を使用する.(バイアス項を除く)
        upperNum = int((self.numHidden-1) * 0.6)
        # 0の1次元ベクトルを作成
        m = np.zeros(self.numHidden-1)
        # 上位50%のインデックスを1にする.
        m[valIndex[0:upperNum]] = 1
        # バイアス項は必ず使うので,[0]番目に1としてマスクする.
        m = np.insert(m,0,1)
        # 使用回数をカウントする
        self.hiddenCount += m

        return m

if __name__ == "__main__":
    """XORの学習テスト"""
    """
    mlp = MultiLayerPerceptron(2, 2, 1, "tanh", "sigmoid")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    mlp.fit(X, y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print i, mlp.predict(i)
    """
    mlp = MultiLayerPerceptron(2, 10, 1, "tanh", "sigmoid")
    print mlp.makeMask()
