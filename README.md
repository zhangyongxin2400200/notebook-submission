这是关于cnn区分bbh的二分类模型以及resnet18区分bns的二分类模型
关于bns，本来我想自己尝试做的不同信噪比的信号（标签为1）和不含信号的（标签为0）但是放到模型中之后并不能很好的区分两者，预测的概率基本都是0.5.然后我就改了data bbh文件，把其中的一些数据改成了适合bns的（包括参数的取值范围，引入了自旋参数，更换模板模型，调整最低频率等）
因为关于这方面的知识我是比较欠缺的，所以这只是我的一些个人拙见，上交的作业中肯定还有一些不合理的地方，欢迎各位老师的批评和指正。
