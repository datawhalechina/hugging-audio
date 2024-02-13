# 补充阅读和资源

本单元提供了语音识别的实践介绍，语音识别是音频领域最受欢迎的任务之一。还想了解更多吗？在这里，您可以找到更多资源，帮助您加深对主题的理解，提升学习体验。

+ Jong Wook Kim 的 [Whisper](https://www.youtube.com/watch?v=fZMiD8sDzzg) 讲座：由 Whisper 的作者 Jong Wook Kim 介绍 Whisper 模型，解释其动机、架构、训练和结果。
+ [端到端语音基准 (ESB)](https://arxiv.org/abs/2210.13352)：一篇全面论证使用正字法 WER 而非标准化 WER 评估 ASR 系统的论文，并介绍了相应的基准
+ [为多语种 ASR 微调 Whisper](https://huggingface.co/blog/fine-tune-whisper)：一篇深入的博文，更详细地解释了 Whisper 模型的工作原理，以及特征提取器和tokenzier所涉及的前处理和后处理步骤
+ [微调多语种 ASR 的 MMS 适配器模型](https://huggingface.co/blog/mms_adapters)：微调 Meta AI 新 [MMS](https://ai.meta.com/blog/multilingual-model-speech-recognition/) 语音识别模型的端到端指南，冻结基础模型权重，仅微调少量适配层
+ [在 🤗 Transformers 中用 n-grams 提升 Wav2Vec2](https://huggingface.co/blog/wav2vec2-with-ngram)：一篇博文，介绍如何将 CTC 模型与外部语言模型 (LM) 相结合，以消除拼写和标点符号错误