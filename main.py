#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import mkdir
from os.path import abspath, dirname, isfile, isdir
from datetime import date, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.random import permutation
import nn
from progressbar import ProgressBar
import time

def scaled(df, feature_columns):
    """
    正規化
    :param df: 正規化前のデータフレーム
    :param not_feature_column :
    :return: 正規化後のデータフレーム
    """
    # データの正規化、かつ少数第三位まで
    ss = StandardScaler()
    for column in feature_columns:
        df[column] = [round(d, 3) for d in ss.fit_transform(df[column].astype(float))]

    return df

def under_sampling(df, label_name="label"):
    # TODO : 交差検定をニューラルネットワークでは出来ない？
    """
    不均衡データの調整
    :param df: 不均衡データフレーム
    :param label_name: 目的変数が格納されているカラム名
    :return: 均衡データフレーム
    """
    # 目的変数に応じて、データを分ける
    many   = df[df[label_name] == 0]
    little = df[df[label_name] == 1]

    # 大小正しい方の変数に格納
    if len(many) < len(little):
        many, little = little, many

    # ループ数を確定
    loop_count = len(many) / len(little)

    # ランダムに並び替え
    many = many.reindex(permutation(many.index))

    little_length = len(little)
    begin = little_length
    end = little_length * 2

    return pd.concat([little, many.iloc[begin:end, :]], axis=0)

    # 少ない方の要素数
    # little_length = len(little)
    # for n in range(loop_count):
    #     begin = little_length * n
    #     end   = little_length * (n + 1)
    #
    #     # 多い方から少ない方の長さ分抽出し、少ない方と結合
    #     yield pd.concat([little, many.iloc[begin:end, :]], axis=0)

def train(net, target_date):
    base_path = abspath(dirname(__file__))
    fname = "%s/data/bks_appstore_train_%s.csv" % (base_path, target_date.strftime("%Y%m%d"))

    # データ読み込み
    df = pd.read_csv(fname)

    # 正規化
    feature_columns = [c for c in df.columns if c not in ["user_profile_id", "label"]]
    scaled_df = scaled(df[(df["retention"] > 1) & (df["retention"] <= 60)], feature_columns)

    # アンダーサンプリング
    us_scl_df = under_sampling(scaled_df)

    # print "us_scl_df_count : ", len(us_scl_df.index)
    p = ProgressBar(len(us_scl_df.index))
    p_count = 0
    for i, row in us_scl_df.iterrows():
        net.train_query(row[feature_columns], [0, 1], row["label"])
        p.update(p_count)
        # print i, p_count
        p_count += 1
    # us_scl_df.map(lambda d: net.train_query(d[feature_columns], [0, 1], d["label"]))

    # 教師データを使って学習
    # for df in us_scl_df.iterrow():
    #     features =
    #     net.train_query(us_scl_df, [0, 1], )

def main():
    # 初期設定
    mynet = nn.SearchNet("drop_predict.db")
    mynet.make_tables()

    start_date = date(2015,5,20)
    end_date = date(2015,5,21)
    for i in range((end_date-start_date).days+1):
        target_date = start_date + timedelta(days=i)
        train(mynet, target_date)

#===================================
# メイン
#===================================
if __name__ == "__main__":
    main()