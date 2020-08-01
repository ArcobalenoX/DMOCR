detection network is based on
 [Advanced_EAST](https://github.com/huoyijie/AdvancedEAST)

对于QUAD 标注框，计算每个有正分数的像素与文本框4个顶点的坐标偏移。
对EAST的输出层结构进行修改，一个通道预测该像素是否在文本框内；
两个通道预测该像素是文本框的头像素还是尾像素；
四个通道预测头像素或尾像素对应的距离两个顶点坐标的偏移量。

 损失函数仍使用的是EAST算法的类平衡交叉熵损失和SmoothL1损失。


#1. preprocess.py
#2. labely.py
#3. train.py
#4. predict.py


