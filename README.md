# doc2vecNegativeSampling
[doc2vec](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)を、gensim等のライブラリを使用せず、negative samplingで実装しました。

データは[tensorflow機械学習クックブック](https://www.amazon.co.jp/TensorFlow%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%82%AF%E3%83%83%E3%82%AF%E3%83%96%E3%83%83%E3%82%AF-Python%E3%83%99%E3%83%BC%E3%82%B9%E3%81%AE%E6%B4%BB%E7%94%A8%E3%83%AC%E3%82%B7%E3%83%9460-impress-top-gear/dp/4295002003)の
doc2vecの章で使われているデータを使用しています。
作られた分散表現を利用して、感情分析をします。
***

```
Python PV_DM.py
```
<br>
PV_DMで分散表現を作成し、学習データとテストデータに分けます。<br>
PV_DMで作成した学習データとテストデータはPVDM.npzとして保存されます。<br>
作成した分散表現もtempファイル内のpvdm.npzに保存されます。<br>


***
```
Python PV_DBOW.py
```
<br>

PV_DBOWで分散表現を作成し、学習データとテストデータに分けます。

PV_DBOWで作成した学習データとテストデータはPV_DBOW.npzとして保存されます。

作成した分散表現もtempファイル内のpvdbow.npzに保存されます。
