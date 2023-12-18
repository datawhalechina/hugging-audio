# 使用流水线进行自动语音识别
自动语音识别（ASR）是一项将语音音频转录为文本的任务。这项任务有许多实际应用，从为视频创建隐藏式字幕到为 Siri 和 Alexa 等虚拟助手启用语音命令。

在本节中，我们将使用`automatic-speech-recognition`流水线转录一段录音，录音内容是一个人询问有关支付账单的问题，使用的数据集与之前的 MINDS-14 相同。

要开始使用，请加载数据集，并按照 "[使用流水线进行音频分类](./audio_classification_with_a_pipeline.md)"中的描述将其升频至 16kHz（如果您还没有这样做的话）。

要转录录音，我们可以使用 Hugging Face `Transformers` 中的自动语音识别流水线。让我们实例化流水线：

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition")
```

接下来，我们将从数据集中提取一个示例，并将其原始数据传递给流水线：

```python
example = minds[0]
asr(example["audio"]["array"])
```

输出：

```python
{"text": "I WOULD LIKE TO PAY MY ELECTRICITY BILL USING MY COD CAN YOU PLEASE ASSIST"}
```

让我们将此输出与此示例的实际转录进行比较：

```python
example["english_transcription"]
```

输出：

```python
"I would like to pay my electricity bill using my card can you please assist"
```

该模型在转录音频方面似乎做得不错！与原始转录相比，它只错了一个单词("card")，考虑到说话者是澳大利亚口音，字母 "r"通常是无声的，这已经很不错了。尽管如此，我还是不建议你用鳕鱼(cod)来支付下一次电费！

默认情况下，该管道使用的是经过自动语音识别训练的英语模型，在本例中没有问题。如果你想尝试用不同语言转录 MINDS-14 的其他子集，可以在 [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=fr&sort=downloads) 上找到预先训练好的 ASR 模型。您可以先按任务过滤模型列表，然后再按语言过滤。找到喜欢的模型后，将其名称作为模型参数传递给流水线。

让我们尝试一下 MINDS-14 的德语分集。加载 "de-DE "子集：

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

找一个例子，看看转录应该是什么样的：

```python
example = minds[0]
example["transcription"]
```

输出：

```python
"ich möchte gerne Geld auf mein Konto einzahlen"
```

在 Hugging Face Hub 上查找预先训练好的德语 ASR 模型，实例化流水线，并转录示例：

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="maxidl/wav2vec2-large-xlsr-german")
asr(example["audio"]["array"])
```

输出：

```python
{"text": "ich möchte gerne geld auf mein konto einzallen"}
```

没错，就是这样！

在解决您自己的任务时，从我们在本单元中展示的简单流水线开始是一个非常有价值的工具，它能带来多种好处：

+ 预训练的模型可能已经很好地解决了你的任务，为你节省了大量时间
+ `pipeline()` 会为你处理所有的前/后处理，所以你不必担心如何将数据转换成适合模型的格式
+ 如果结果并不理想，这仍然能为您提供一个快速的基准，以便将来进行微调
+ 一旦您在自定义数据上对模型进行了微调并在 Hub 上进行了分享，整个社区就可以通过 `pipeline()` 方法快速、轻松地使用该模型，从而使人工智能更易于使用。