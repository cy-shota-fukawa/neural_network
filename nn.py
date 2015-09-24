#!/usr/bin/python
# -*- coding: utf-8 -*-
from math import tanh
import sqlite3 as sqlite

def dtanh(y):
    """
    tanhの傾きを取得。
    出力が0.0辺りだと傾きが急勾配であるため、入力をほんのすこし変更しただけでも大きな影響がある。
    出力が-1・1に近づくに連れ入力の変更が出力に及ぼす影響は小さくなっていく。
    """
    return 1.0 - y * y

class SearchNet:
    """
    多層パーセプトロンネットワーク
    """
    def __init__(self, dbname):
        self.con    = sqlite.connect(dbname)

    def __del__(self):
        self.con.close()

    def back_propagate(self, targets, N=0.5):
        """
        1. ノードの現在の出力とあるべき出力の差を計算する。
        2. dtanh関数を使ってノードの入力の合計をどれくらい変更すべきかを決める。
        3. 入ってくるリンクすべての強度を、リンクの現在の強度と学習率に見合うよう変更する。
        """
        # 出力の誤差を計算する
        output_deltas = [0.0] * len(self.urlids)
        for k in range(len(self.urlids)):
            # 現在の出力とあるべき出力の差を計算する
            error = targets[k] - self.ao[k]

            # ノードの入力の合計をどれくらい変更すべきかを決める
            output_deltas[k] = dtanh(self.ao[k]) * error

        # 隠れ層の誤差を計算する
        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range(len(self.hiddenids)):
            error = 0.0
            for k in range(len(self.urlids)):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # 出力の重みを更新する
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change

        # 入力の重みを更新する
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hidden_deltas[j] * self.ai[i]
            self.wi[i][j] = self.wi[i][j] + N*change

    def train_query(self, features, label_type, label):
        """
        :param features: 説明変数
        :param label_type: 目的変数の種類(0,1)
        :param label: 目的変数
        :return: なし
        """
        # 必要であればhidden nodeを生成する
        self.generate_hiddennode(features, label_type)
        self.setup_network(features, label_type)
        self.feed_forward()
        targets = [0.0] * len(label_type)
        targets[label_type.index(label)] = 1.0
        error = self.back_propagate(targets)
        self.update_database()

    def update_database(self):
        """
        インスタンス変数のwiとwoに保存されている新たな重みでデータベースを更新
        """
        # データベースの値にセットする
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                self.set_strength(self.wordids[i], self.hiddenids[j], 0, self.wi[i][j])

        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                self.set_strength(self.hiddenids[j], self.urlids[k], 1, self.wo[j][k])

        self.con.commit()

    def make_tables(self):
        # TODO: 速度に問題があるようであれば、indexをはる
        self.con.execute("CREATE TABLE hiddennode(create_key)")
        self.con.execute("CREATE TABLE wordhidden(fromid, toid, strength)")
        self.con.execute("CREATE TABLE hiddenurl(fromid, toid, strength)")

        self.con.execute("CREATE INDEX hiddenidx on hiddennode(create_key)")
        self.con.execute("CREATE INDEX wordfromidx on wordhidden(fromid)")
        self.con.execute("CREATE INDEX wordtoidx on wordhidden(toid)")
        self.con.execute("CREATE INDEX urlfromidx on hiddenurl(fromid)")
        self.con.execute("CREATE INDEX urltoidx on hiddenurl(toid)")

        self.con.commit()

    def get_strength(self, fromid, toid, layer):
        """コネクションの強さを取得
        コネクションが存在しない場合はデフォルト値を返す
        layer 0: 単語同士のコネクション
        layer 1: URL同士のコネクション
        """
        if layer == 0: table = "wordhidden"
        else: table = "hiddenurl"

        res = self.con.execute("SELECT strength FROM %s WHERE fromid = %d AND toid = %d" % (table, fromid, toid)).fetchone()

        if res == None:
            if layer == 0: return -0.2
            if layer == 1: return 0

        return res[0]

    def set_strength(self, fromid, toid, layer, strength):
        """
        コネクションの有無を確認し、存在すれば新たな強度でコネクションを更新、存在しなければコネクションを作成
        layer 0: 単語同士のコネクション
        layer 1: URL同士のコネクション
        """
        if layer == 0: table = "wordhidden"
        else: table = "hiddenurl"

        res = self.con.execute("SELECT rowid FROM %s WHERE fromid = %d AND toid = %d" % (table, fromid, toid)).fetchone()

        if res == None:
            self.con.execute("INSERT INTO %s (fromid, toid, strength) VALUES (%d, %d, %f)" % (table, fromid, toid, strength))
        else:
            rowid = res[0]
            self.con.execute("UPDATE %s SET strength = %f WHERE rowid = %d" % (table, strength, rowid))

    def generate_hiddennode(self, wordids, urls):
        """
        登録されていない単語の組み合わせが渡されるたびに隠れ層に新たなノードを作り、デフォルトの重みでつなぎ合わせる。
        クエリのノードとクエリによって返されるURLの間もデフォルトの重みでつなぎ合わせる。
        """
        # この単語セットに対してノードを既に作り上げているか調べる
        create_key = "_".join([str(wi) for wi in wordids])
        res = self.con.execute("SELECT rowid FROM hiddennode WHERE create_key='%s'" % create_key).fetchone()

        # もしノードがなれけば作る
        if res == None:
            cur = self.con.execute("INSERT INTO hiddennode (create_key) VALUES ('%s')" % create_key)
            hiddenid = cur.lastrowid

            # 何らかのデフォルト値をセットする
            for wordid in wordids:
                self.set_strength(wordid, hiddenid, 0, 1.0 / len(wordids))
            for urlid in urls:
                self.set_strength(hiddenid, urlid, 1, 0.1)
            self.con.commit()

    def get_tall_hidden_ids(self, wordids, urlids):
        """
        特定のクエリに関する隠れ層の全てのノードを探し出す
        """
        l1 = {}
        for wordid in wordids:
            cur = self.con.execute("SELECT toid FROM wordhidden WHERE fromid = %d" % wordid)
            for row in cur: l1[row[0]] = 1
        for urlid in urlids:
            cur = self.con.execute("SELECT fromid FROM hiddenurl WHERE toid = %d" % urlid)
            for row in cur: l1[row[0]] = 1
        return l1.keys()

    def setup_network(self, wordids, urlids):
        """
        データベースから引き出した現在のすべての重みでネットワークを構築する
        """
        # 値のリスト
        self.wordids = wordids
        self.hiddenids = self.get_tall_hidden_ids(wordids, urlids)
        self.urlids = urlids

        # ノードの出力
        self.ai = [1.0] * len(self.wordids)
        self.ah = [1.0] * len(self.hiddenids)
        self.ao = [1.0] * len(self.urlids)

        # 重みの行列を作る
        self.wi = [[self.get_strength(wordid, hiddenid, 0)
                        for hiddenid in self.hiddenids]
                        for wordid in self.wordids]

        self.wo = [[self.get_strength(hiddenid, urlid, 1)
                        for urlid in self.urlids]
                        for hiddenid in self.hiddenids]

    def feed_forward(self):
        """
        フィードフォワードアルゴリズム
        """
        # 入力はクエリの単語たち？(aiの初期化)
        for i in range(len(self.wordids)):
            self.ai[i] = 1.0

        # 隠れ層の発火
        for j in range(len(self.hiddenids)):
            sum = 0.0
            for i in range(len(self.wordids)):
                # リンクの強度を掛け合わせる
                # TODO : なぜaiを使うのか。1.0直値ではいけない理由が不明
                # sum = sum + self.ai[j] * self.wi[i][j]
                sum = sum + 1.0 * self.wi[i][j]
            # tanhを適用して最終的な出力を作り出す
            self.ah[j] = tanh(sum)

        # 出力層の発火
        for k in range(len(self.urlids)):
            sum = 0.0
            for j in range(len(self.hiddenids)):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def get_result(self, features, label_type):
        """
        結果の取得
        :param features: 説明変数
        :param label_type: 目的変数の種類(0, 1)
        :return: 結果
        """
        self.setup_network(features, label_type)
        return self.feed_forward()