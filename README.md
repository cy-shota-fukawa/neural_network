+ SVM版スクリプト
    + 問題点
        + 何のパラメータを上げればよいかが日によってバラバラ
            + 入力によってパラメータの影響度合いが変わる
            + 入力は1日分のデータ。その日の特徴にあわせてモデルが作られることになる
        + 追加学習が出来ない

    + 改善
        + 教師データを数日分に
    + 結果
        + 少しはよくなった

    + 20150709 今後
        + 追加学習をするのが良いとは限らない
        + 主成分分析
            + 将来的に考えると次元削減の方がよい
        + ヒューリスティック
            + いらないと思われるデータを消す
        + シナイベの開催期間
        + ログボを上げるくらいの施策であれば複数パターンがあっても良いのではないか

+ SVMだと追加の学習が出来ない
    +

+ nn.py
    + 予測
        + 説明変数と目的変数の種類を渡し、目的変数の種類毎に確立を返す
    + 変数
        + ai : 説明変数の数
        + ah : 隠れ層のノード数
        + ao : 目的変数の種類数

+ Progressbar
    + インストール
        + sudo pip install progressbar2