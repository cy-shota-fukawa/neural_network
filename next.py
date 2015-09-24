#!/usr/bin/python
# -*- coding: utf-8 -*-
import nn

#===================================
# メイン
#===================================
if __name__ == "__main__":
    mynet = nn.SearchNet("nn.db")
    wWorld, wRiver, wBank = 101, 102, 103
    uWorldBank, uRiver, uEarth = 201, 202, 203

    user_data = {
        "A":{
            "feature":[0, 1000, 4, 32],
            "label":0
        },
        "B":{
            "feature":[100, 10000, 5, 360],
            "label":1
        },
        "C":{
            "feature":[0, 2000, 4, 60],
            "label":0
        },
        "D":{
            "feature":[0, 11000, 4, 300],
            "label":1
        }
    }

    output_data = [0, 1]
    for id, data in user_data.items():
        mynet.train_query(data["feature"], output_data, data["label"])

    test_data = [
        [0, 1000, 0, 1000],
        [100, 0, 2, 10],
        [0, 0, 0, 0],
        [100, 100, 100, 100]
    ]
    for td in test_data:
        print mynet.get_result(td, output_data)