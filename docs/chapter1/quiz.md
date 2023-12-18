# 检查您对教程的理解

1. 采样率用什么单位衡量？

- ( ) 分贝
- (x) 赫兹
- ( ) 比特

2. 流式传输大型音频数据集时，多长时间可以开始使用？

- ( )下载完整数据集后立即开始
- ( ) 下载前 16 个示例后立即开始
- (x) 第一个示例下载完成后

3. 什么是时频谱？

- ( ) 用于将麦克风首先捕捉到的音频数字化的设备，麦克风将声波转换为电信号
- ( ) 显示音频信号振幅随时间变化的曲线图。它也被称为声音的时域表示法。
- (x) 信号随时间变化的频谱直观表示

4. 将原始音频数据转换成 Whisper 所期望的对数梅尔频谱图的最简单方法是什么？

- ( ) `librosa.feature.melspectrogram(audio["array"])`
- (x) `feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")` ; `feature_extractor(audio["array"])`
- ( ) `dataset.feature(audio["array"], model="whisper")`

5. 如何从Hugging Face Hub 载入数据集？

+ (x) `from datasets import load_dataset` ; `dataset = load_dataset(DATASET_NAME_ON_HUB)`
+ ( ) `import librosa `; `dataset = librosa.load(PATH_TO_DATASET)`
+ ( ) `from transformers import load_dataset` ; `dataset = load_dataset(DATASET_NAME_ON_HUB)`

6. 您的自定义数据集包含采样率为 32 kHz 的高质量音频。您想训练一个语音识别模型，该模型希望音频示例的采样率为 16 kHz。该怎么办？

- ( ) 按原样使用示例，模型将很容易泛化到更高质量的音频示例
- (x) 使用 Hugging Face `Datasets` 库中的音频模块对自定义数据集中的示例进行低采样
- ( ) 通过丢弃每一个其他样本，将采样率降低 2 倍

7. 如何将机器学习模型生成的时频谱转换成波形？

+ (x) 我们可以使用一个名为声码器的神经网络，从时频谱中重建波形
+ ( ) 我们可以使用反 STFT 将生成的频谱图转换为波形
+ ( ) 您无法将机器学习模型生成的频谱图转换成波形