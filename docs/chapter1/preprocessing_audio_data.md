# 预处理音频数据集

用Hugging Face `Datasets`加载数据集只是乐趣的一半。如果您计划将其用于训练模型或运行推理，则需要先对数据进行预处理。一般来说，这将涉及以下步骤：

+ 对音频数据重新采样
+ 过滤数据集
+ 将音频数据转换为模型的预期输入

## 对音频数据重新采样
`load_dataset` 函数以发布时的采样率下载音频示例。这并不总是您计划训练或用于推理的模型所期望的采样率。如果采样率之间存在差异，可以按照模型的预期采样率对音频重新采样。

大多数可用的预训练模型都是在采样率为 16 kHz 的音频数据集上进行预训练的。当我们探索 MINDS-14 数据集时，您可能已经注意到它的采样率为 8 kHz，这意味着我们可能需要对其进行上采样。

为此，请使用 Hugging Face `Datasets`的 `cast_column` 方法。此操作不会就地更改音频，而是在加载音频示例时向数据集发出信号，让数据集即时重新采样。以下代码将把采样率设置为 16kHz：

```python
from datasets import Audio

minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

重新加载 MINDS-14 数据集中的第一个音频示例，检查其是否已重新采样到所需的采样率：

```python
minds[0]
```

输出：

```python
{
    "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
    "audio": {
        "path": "/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-AU~PAY_BILL/response_4.wav",
        "array": array(
            [
                2.0634243e-05,
                1.9437837e-04,
                2.2419340e-04,
                ...,
                9.3852862e-04,
                1.1302452e-03,
                7.1531429e-04,
            ],
            dtype=float32,
        ),
        "sampling_rate": 16000,
    },
    "transcription": "I would like to pay my electricity bill using my card can you please assist",
    "intent_class": 13,
}
```

您可能会注意到，现在的数组值也不同了。这是因为我们现在每个振幅值的数量都是以前的两倍。

>  :bulb: 关于重采样的一些背景知识： 如果音频信号的采样频率为 8 kHz，即每秒有 8000 个采样读数，那么我们就知道音频不包含任何超过 4 kHz 的频率。这是奈奎斯特采样定理所保证的。正因为如此，我们可以确定，在采样点之间，原始的连续信号始终是一条平滑的曲线。将采样率提高到更高的采样率，只需通过近似这条曲线，计算出在现有采样点之间的额外采样值即可。而降低采样率则要求我们在估算新的采样点之前，首先滤除任何高于新奈奎斯特极限的频率。换句话说，你不能通过简单地丢弃所有其他采样点来将采样率降低 2 倍--这会在信号中产生失真，即所谓的 "混叠"。正确进行重采样非常棘手，最好交由 `librosa` 或 Hugging Face `Datasets` 等久经考验的库处理。

## 过滤数据集
您可能需要根据某些标准过滤数据。常见的情况之一是将音频示例限制在一定的持续时间内。例如，我们可能想要过滤掉任何超过 20 秒的示例，以防止在训练模型时出现内存不足的错误。

我们可以使用 Hugging Face 数据集的过滤器方法，并向其传递一个包含过滤逻辑的函数。让我们先编写一个函数，指明哪些示例要保留，哪些要丢弃。这个函数 `is_audio_length_in_range` 会在样本短于 20 秒时返回 True，在样本长于 20 秒时返回 False。

```python
MAX_DURATION_IN_SECONDS = 20.0


def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS
```

过滤功能可应用于数据集的列，但我们的数据集中没有包含音轨持续时间的列。不过，我们可以创建一列，根据该列中的值进行过滤，然后将其移除。

```python
# 使用 librosa 从音频文件中获取示例的持续时间
new_column = [librosa.get_duration(path=x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)

# 使用 Hugging Face Datasets 的 `filter` 方法应用过滤功能
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

# 移除临时辅助列
minds = minds.remove_columns(["duration"])
minds
```

输出：

```python
Dataset({features: ["path", "audio", "transcription", "intent_class"], num_rows: 624})
```

我们可以确认，数据集已从 654 个实例筛选为 624 个。

## 预处理音频数据
处理音频数据集最具挑战性的方面之一，就是以正确的格式为模型训练准备数据。正如您所看到的，原始音频数据是一个样本值数组。然而，预训练的模型，不管是用于推理，还是想针对任务进行微调，都需要将原始数据转换为输入特征。不同模型对输入特征的要求可能会有所不同--这取决于模型的架构和预训练的数据。好消息是，对于每个支持的音频模型，Hugging Face `Transformers` 都提供了一个特征提取器类，可以将原始音频数据转换为模型所需的输入特征。

那么，特征提取器如何处理原始音频数据呢？让我们看看 [Whisper](https://arxiv.org/abs/2212.04356) 的特征提取器，了解一些常见的特征提取转换。Whisper 是由来自 OpenAI 的 Alec Radford 等人于 2022 年 9 月发布的用于自动语音识别（ASR）的预训练模型。

首先，Whisper 特征提取器对一批音频示例进行填充(pad)/截断(truncate)，使所有示例的输入长度为 30 秒。短于 30 秒的示例通过在序列末尾添加 0 来填充到 30 秒（音频信号中的 0 相当于无信号或静音）。长于 30 秒的样本会被截断为 30 秒。由于批次中的所有元素都被填充/截断到输入空间的最大长度，因此不需要attention mask。Whisper 在这一点上是独一无二的，其他大多数音频模型都需要一个attention mask，详细说明序列在哪些地方被填充，因此在自注意力机制中哪些地方应被忽略。Whisper 经过训练，可以在没有attention mask的情况下工作，并直接从语音信号中推断出忽略输入的位置。

Whisper 特征提取器执行的第二项操作是将填充音频阵列转换为对数梅尔时频谱。正如你所记得的，这些时频谱描述了信号频率随时间的变化情况，用梅尔标度表示，用分贝（对数部分）测量，使频率和振幅更能代表人的听觉。

只需几行代码，就能对原始音频数据进行所有这些转换。让我们继续从预训练的 Whisper 检查点加载特征提取器，为我们的音频数据做好准备：

```python
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
```

接下来，您可以编写一个函数，通过`feature_extractor`对单个音频示例进行预处理。

我们可以使用 Hugging Face `Datasets`的`map`方法，将数据准备功能应用于所有训练示例：

```python
minds = minds.map(prepare_dataset)
minds
```

输出：

```python
Dataset(
    {
        features: ["path", "audio", "transcription", "intent_class", "input_features"],
        num_rows: 624,
    }
)
```

就这么简单，我们现在就可以把对数梅尔频谱作为数据集中的输入特征。

让我们对`minds`数据集中的一个例子进行可视化：

```python
import numpy as np

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()
```

![对数梅尔频谱](images/log_mel_whisper.png)

现在你可以看到经过预处理后的 Whisper 模型音频输入。

模型的特征提取器类负责将原始音频数据转换为模型所需的格式。然而，许多涉及音频的任务都是多模态的，例如语音识别。在这种情况下，Hugging Face `Transformers`还提供特定于模型的tokenizers来处理文本输入。

你可以为 Whisper 和其他多模态模型分别加载特征提取器和tokenizer，也可以通过所谓的处理器加载两者。为了让事情变得更简单，可以使用自动处理器（`AutoProcessor`）从检查点加载模型的特征提取器和处理器，就像下面这样：

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-small")
```

我们在此说明了基本的数据准备步骤。当然，自定义数据可能需要更复杂的预处理。在这种情况下，您可以扩展函数 `prepare_dataset`，以执行任何类型的自定义数据转换。有了 Hugging Face 数据集，只要你能把它写成一个 Python 函数，就能把它应用到你的数据集中！