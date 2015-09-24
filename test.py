#!/usr/bin/python
# -*- coding: utf-8 -*-
import nn

#===================================
# メイン
#===================================
if __name__ == "__main__":
    # 初期設定
    mynet = nn.SearchNet("nn.db")
    mynet.make_tables()

    # サンプル変数定義
    wWorld, wRiver, wBank = 101, 102, 103
    uWorldBank, uRiver, uEarth = 201, 202, 203

    # point, gold, fq, retentionと仮定
    user_data = {
        "A":[0, 1000, 4, 32],
        "B":[100, 10000, 5, 360],
        "C":[0, 2000, 4, 60],
        "D":[0, 11000, 4, 300]
    }

    output_data = [0, 1]

    # 全てのノードの組み合わせをDBに登録する
    # TODO:使い方があっているのかの確認
    # for user_no in range(len(user_max))
    for id, data in user_data.items():
        print id, data
        mynet.generate_hiddennode(data, output_data)
    # mynet.generate_hiddennode([wWorld, wBank], [uWorldBank, uEarth, uWorldBank])
    for c in mynet.con.execute("SELECT * FROM wordhidden"): print c
    for c in mynet.con.execute("SELECT * FROM hiddenurl"): print c

    print mynet.get_result(wordids=user_data["A"], urlids=output_data)