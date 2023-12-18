# 音频数据流
音频数据集面临的最大挑战之一是其庞大的体积。一分钟未压缩的 CD 音质音频（44.1kHz，16 位）的存储空间略大于 5 MB。通常情况下，一个音频数据集会包含数小时的录音。

在前面的章节中，我们使用了 MINDS-14 音频数据集的一个很小的子集，然而，典型的音频数据集要大得多。例如，SpeechColab 的 GigaSpeech 的 xs（最小）配置仅包含 10 小时的训练数据，但下载和准备却需要超过 13GB 的存储空间。那么，如果我们想在更大的分割数据上进行训练，会发生什么情况呢？同一数据集的完整 xl 配置包含 10,000 小时的训练数据，需要超过 1TB 的存储空间。对于我们大多数人来说，这远远超出了普通硬盘的规格。我们需要花钱购买额外的存储空间吗？还是说我们有办法在没有磁盘空间限制的情况下对这些数据集进行训练？

Hugging Face `Datasets` 通过提供流模式来拯救我们。流模式允许我们在迭代数据集时逐步加载数据。我们不是一次性下载整个数据集，而是一次加载一个示例。我们对数据集进行迭代，在需要时即时加载和准备示例。这样，我们只会加载我们正在使用的示例，而不会加载我们不需要的示例！完成一个示例后，我们会继续迭代数据集并加载下一个示例。

与一次性下载整个数据集相比，流模式有三个主要优势：

+ 磁盘空间：当我们遍历数据集时，示例会逐个加载到内存中。由于数据不是下载到本地，因此不需要磁盘空间，因此可以使用任意大小的数据集。
+ 下载和处理时间：音频数据集很大，下载和处理需要大量时间。而使用流媒体时，加载和处理都是即时完成的，这意味着只要第一个示例准备就绪，您就可以开始使用数据集。
+ 便于实验：您可以在少量示例上进行实验，检查脚本是否正常运行，而无需下载整个数据集。

流模式有一个注意事项。在不使用流模式下载完整数据集时，原始数据和处理过的数据都会保存到本地磁盘。如果我们想重新使用该数据集，可以直接从磁盘加载处理过的数据，跳过下载和处理步骤。因此，我们只需执行一次下载和处理操作，之后就可以重新使用准备好的数据。

在流模式下，数据不会下载到磁盘。因此，下载的数据和预处理的数据都不会被缓存。如果我们想重新使用数据集，就必须重复流式传输步骤，重新加载音频文件并进行处理。因此，建议您下载可能多次使用的数据集。

如何启用流模式？很简单！只需在加载数据集时设置 `streaming=True`，其余的事情都会有人帮你处理。

```python
gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
```

就像我们对下载的 MINDS-14 子集进行预处理一样，您也可以以完全相同的方式对流数据集进行同样的预处理。

唯一不同的是，您不能再使用 Python 索引来访问单个样本（即：`gigaspeech["train"][sample_idx]`）。相反，您必须遍历数据集。以下是流式传输数据集时访问示例的方法：

```python
next(iter(gigaspeech["train"]))
```

输出：

```python
{
    "segment_id": "YOU0000000315_S0000660",
    "speaker": "N/A",
    "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
    "audio": {
        "path": "xs_chunks_0000/YOU0000000315_S0000660.wav",
        "array": array(
            [0.0005188, 0.00085449, 0.00012207, ..., 0.00125122, 0.00076294, 0.00036621]
        ),
        "sampling_rate": 16000,
    },
    "begin_time": 2941.89,
    "end_time": 2945.07,
    "audio_id": "YOU0000000315",
    "title": "Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43",
    "url": "https://www.youtube.com/watch?v=zr2n1fLVasU",
    "source": 2,
    "category": 24,
    "original_full_path": "audio/youtube/P0004/YOU0000000315.opus",
}
```

如果您想预览大型数据集中的多个示例，可使用 `take()` 获取前 n 个元素。让我们抓取 gigaspeech 数据集中的前两个示例：

```python
gigaspeech_head = gigaspeech["train"].take(2)
list(gigaspeech_head)
```

输出：

```python
[
    {
        "segment_id": "YOU0000000315_S0000660",
        "speaker": "N/A",
        "text": "AS THEY'RE LEAVING <COMMA> CAN KASH PULL ZAHRA ASIDE REALLY QUICKLY <QUESTIONMARK>",
        "audio": {
            "path": "xs_chunks_0000/YOU0000000315_S0000660.wav",
            "array": array(
                [
                    0.0005188,
                    0.00085449,
                    0.00012207,
                    ...,
                    0.00125122,
                    0.00076294,
                    0.00036621,
                ]
            ),
            "sampling_rate": 16000,
        },
        "begin_time": 2941.89,
        "end_time": 2945.07,
        "audio_id": "YOU0000000315",
        "title": "Return to Vasselheim | Critical Role: VOX MACHINA | Episode 43",
        "url": "https://www.youtube.com/watch?v=zr2n1fLVasU",
        "source": 2,
        "category": 24,
        "original_full_path": "audio/youtube/P0004/YOU0000000315.opus",
    },
    {
        "segment_id": "AUD0000001043_S0000775",
        "speaker": "N/A",
        "text": "SIX TOMATOES <PERIOD>",
        "audio": {
            "path": "xs_chunks_0000/AUD0000001043_S0000775.wav",
            "array": array(
                [
                    1.43432617e-03,
                    1.37329102e-03,
                    1.31225586e-03,
                    ...,
                    -6.10351562e-05,
                    -1.22070312e-04,
                    -1.83105469e-04,
                ]
            ),
            "sampling_rate": 16000,
        },
        "begin_time": 3673.96,
        "end_time": 3675.26,
        "audio_id": "AUD0000001043",
        "title": "Asteroid of Fear",
        "url": "http//www.archive.org/download/asteroid_of_fear_1012_librivox/asteroid_of_fear_1012_librivox_64kb_mp3.zip",
        "source": 0,
        "category": 28,
        "original_full_path": "audio/audiobook/P0011/AUD0000001043.opus",
    },
]
```

流模式可以让您的研究更上一层楼：您不仅可以访问最大的数据集，还可以轻松地一次性评估多个数据集上的系统，而不必担心磁盘空间。与单个数据集评估相比，多数据集评估能更好地衡量语音识别系统的泛化能力（如端到端语音基准 (ESB)）。