# 评论文本情感分析

应用了停用词表和朴素贝叶斯模型，通过交叉验证提高模型准确度，对用户评论进行情感分析

## 使用停用词表对评论做预处理，省去训练时间和减少模型影响

data.csv为美团外卖的星级评分与用户评论
comments.csv为携程景点的评分和用户评论

## How to run?

` python main.py `

## TODOS

- 融合句法规则，尝试将朴素贝叶斯模型与词典结合应用

[参考文献](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2019&filename=SJSJ201911047&uniplatform=NZKPT&v=7d893uLcWt7NrzxIRVtHOwOIYK6gq_dsmbMPYPptHwxgDVbTqtR1XOhpe1TWrdPF)

