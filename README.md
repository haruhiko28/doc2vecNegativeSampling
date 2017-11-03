# doc2vecNegativeSampling
[doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)を、gensim等のライブラリを使用せず、negative samplingで実装しました。

データは[tensorflow機械学習クックブック](https://www.amazon.co.jp/TensorFlow%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%82%AF%E3%83%83%E3%82%AF%E3%83%96%E3%83%83%E3%82%AF-Python%E3%83%99%E3%83%BC%E3%82%B9%E3%81%AE%E6%B4%BB%E7%94%A8%E3%83%AC%E3%82%B7%E3%83%9460-impress-top-gear/dp/4295002003)の
doc2vecの章で使われているデータを使用しています。
作られた分散表現を利用して、感情分析をします。
***
### PV_DM
<br>
```
Python PV_DM.py
```
<br>
PV_DMで分散表現を作成し、学習データとテストデータに分けます。<br>
PV_DMで作成した学習データとテストデータはPVDM.npzとして保存されます。<br>
作成した分散表現の行列もtempファイル内のpvdm.npzに保存されます。<br>

<br>
***
### PV_DBOW
<br>
```
Python PV_DBOW.py
```
<br>
PV_DBOWで分散表現を作成し、学習データとテストデータに分けます。<br>
作成した学習データとテストデータはPV_DBOW.npzとして保存されます。<br>
作成した分散表現の行列もtempファイル内のpvdbow.npzに保存されます。<br>

***
### PV_DMとPV_DBOW
<br>
```
Python doc2vec.py
```
<br>
上の両方を使って、分散表現を作成し、学習データとテストデータに分けます。<br>
作成された学習データとテストデータはDOUBLE.npzとして保存されます。<br>
作成した分散表現の行列もtempファイル内のmovie.npzに保存されます。<br>
<br>
***
### Kerasで感情分析
<br>
最後にKerasを使って感情分析(positive/negative)を行います。

```
 Python Keras_SentiAna.py DOUBLE 
```
DOUBLE のところをPVDM, PVDBOWに変えれば、PVDMのみ、またはPVDBOWのみで学習した分散表現を使用できます。<br>
#### 実行結果<br>
```
8362/8362 [==============================] - 0s - loss: 0.5421 - acc: 0.7282      
Epoch 2/20
8362/8362 [==============================] - 0s - loss: 0.4426 - acc: 0.8009     
Epoch 3/20
8362/8362 [==============================] - 0s - loss: 0.3874 - acc: 0.8323     
Epoch 4/20
8362/8362 [==============================] - 0s - loss: 0.3407 - acc: 0.8595     
Epoch 5/20
8362/8362 [==============================] - 0s - loss: 0.3028 - acc: 0.8784     
Epoch 6/20
8362/8362 [==============================] - 0s - loss: 0.2701 - acc: 0.8938     
Epoch 7/20
8362/8362 [==============================] - 0s - loss: 0.2429 - acc: 0.9074     
Epoch 8/20
8362/8362 [==============================] - 0s - loss: 0.2183 - acc: 0.9162     
Epoch 9/20
8362/8362 [==============================] - 0s - loss: 0.1958 - acc: 0.9291     
Epoch 10/20
8362/8362 [==============================] - 0s - loss: 0.1771 - acc: 0.9354     
Epoch 11/20
8362/8362 [==============================] - 0s - loss: 0.1615 - acc: 0.9445     
Epoch 12/20
8362/8362 [==============================] - 0s - loss: 0.1431 - acc: 0.9535     
Epoch 13/20
8362/8362 [==============================] - 0s - loss: 0.1296 - acc: 0.9605     
Epoch 14/20
8362/8362 [==============================] - 0s - loss: 0.1129 - acc: 0.9681     
Epoch 15/20
8362/8362 [==============================] - 0s - loss: 0.1008 - acc: 0.9755     
Epoch 16/20
8362/8362 [==============================] - 0s - loss: 0.0887 - acc: 0.9791     
Epoch 17/20
8362/8362 [==============================] - 0s - loss: 0.0800 - acc: 0.9829     
Epoch 18/20
8362/8362 [==============================] - 0s - loss: 0.0700 - acc: 0.9858     
Epoch 19/20
8362/8362 [==============================] - 0s - loss: 0.0617 - acc: 0.9886     
Epoch 20/20
8362/8362 [==============================] - 0s - loss: 0.0546 - acc: 0.9916     
  32/2091 [..............................] - ETA: 1sloss =  0.526023227508
accuracy =  0.823051171688
```
