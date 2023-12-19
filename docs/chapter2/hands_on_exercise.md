# 实践练习
本练习不计分，旨在帮助您熟悉在课程其余部分中将要使用的工具和库。如果您在使用 Google Colab、Hugging Face `Datasets`、`librosa` 和 Hugging Face `Transformers` 方面已经很有经验，可以跳过本练习。

1. 创建 Google Colab 笔记本 / Jupyter Notebook。
2. 使用 Hugging Face Datasets 以流式模式加载 [`facebook/voxpopuli`](https://huggingface.co/datasets/facebook/voxpopuli) 数据集的训练分集，语言自选。
3. 从数据集中的训练部分获取第三个示例并进行探索。根据该示例的特征，您可以将该数据集用于哪些音频任务？
4. 绘制这个示例的波形图和时频谱。
5. 转到 Hugging Face Hub，探索预训练模型，并找到一个可用于自动语音识别的模型，用于您之前选择的语言。用找到的模型实例化相应的流水线，并转录示例。
6. 将从流水线中得到的转录结果与示例中提供的转录结果进行比较。

如果您在此练习中遇到困难，请随时查看[示例解决方案](https://github.com/datawhalechina/hugging-audio/blob/main/codes/chapter2/chapter2_exercise.ipynb)。发现了什么有趣的东西？发现了一个很酷的模型？有漂亮的时频图吗？

在接下来的章节中，你将了解更多有关各种音频Transformer架构的知识，并训练自己的模型！