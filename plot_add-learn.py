#coding:utf-8
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv

argvs = sys.argv
argc = len(argvs)

# シミュレーションで取得したステップ
#template = [5, 10, 50, 100, 500, 1000, 1500, 3000, 5000, 7500, 10000, 15000, 30000, 35000, 50000]
template = [5, 10, 50, 100, 500, 1000, 1500, 3000, 3005, 3010, 3050, 3100, 3500, 4000, 4500, 6000]

if argc < 2:
    print "file not found."
    exit()

# コマンドラインからファイル名を取得
filename = argvs[1]
# 読み込む
data = np.loadtxt(filename, delimiter=",")
data = data[1:]

# 次元数をサンプル数の数と合わせる
file_dim = len(data[1:]) + 1
print file_dim

# x軸のメモリをステップにする
plt.xticks(template[0:file_dim], template[0:file_dim])

# プロットするステップを抽出
ll = template[0:file_dim]
# 10本の折れ線グラフを描画
for i in range(10):
    print i
    plt.plot(ll, data[:,i],"-o")

# 凡例を左に寄せる
#plt.legend(Range(10), loc="left")

# x軸に最大ステップの12分の1を余白にする
x_margin = template[file_dim-1] / 15
# x軸の範囲を最後のステップにする
plt.xlim(-x_margin, template[file_dim-1] + x_margin)

# y軸に0.05の値分の余白を加える
y_margin = 0.05
# y軸のマージンも加える
plt.ylim(-y_margin, 1.0+y_margin)



# y軸のラベル
plt.ylabel("F-score")
# x軸のラベル
plt.xlabel("Step")

#plt.show()

#loc='lower right'で、右下に凡例を表示
plt.legend(range(10), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
# 右側の余白を調整
plt.subplots_adjust(right=0.8)
plt.show()
