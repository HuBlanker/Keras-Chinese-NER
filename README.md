# Keras-Chinese-NER
chinese text ner demo

关于本项目的详细信息请查看 [2019-11-24-用深度学习解决NLP中的命名实体识别(NER)问题(深度学习入门项目)(http://huyan.couplecoders.tech/java/nlp/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/ner/2019/11/24/%E7%94%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%A7%A3%E5%86%B3NLP%E4%B8%AD%E7%9A%84%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB(NER)%E9%97%AE%E9%A2%98(%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%E9%A1%B9%E7%9B%AE)/)

## Use

1. clone项目
2. 构建自己的数据集.
3. 修改`model.py`中的tag列表为自己数据中的tag全集.
4. 执行`python3 model.py --mode=train`.   # 注意参数的默认值,可以自行修改
5. 训练完成之后, 将`predict-online/DemoMain.java`中的代码集成到自己的项目.

## Result

模型预测正确率97%(训练了30个epoch), 线上预测耗时20ms.

## References

完成此项目时, 参考了下面两个项目,在此表示感谢.

[https://github.com/Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF)

[https://github.com/stephen-v/zh-NER-keras](https://github.com/stephen-v/zh-NER-keras)
