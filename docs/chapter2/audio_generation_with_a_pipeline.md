# 使用流水线生成音频
音频生成包含一系列涉及音频输出的多功能任务。我们将在此探讨的任务是语音生成（又称 "文本到语音"）和音乐生成。在 "文本到语音 "过程中，一个模型会将一段文字转换成栩栩如生的口语声音，从而为虚拟助手、视障人士无障碍工具和个性化有声读物等应用打开大门。另一方面，音乐生成可以实现创意表达，主要用于娱乐和游戏开发行业。

在Hugging Face `Transformers` 中，你会发现一个涵盖这两种任务的流水线。这个流水线被称为 "`text_to_audio`"，但为了方便起见，它也有一个 "`text_to_speech` "的别名。在这里，我们将同时使用这两种方法，你可以根据自己的任务选择更适用的方法。

让我们来探讨一下如何使用这一流水线，只需几行代码就能开始为文本和音乐生成音频旁白。

该流水线是Hugging Face Transformers 的新功能，是 4.32 版的一部分。因此，您需要将库升级到最新版本才能获得该功能：

```python
pip install --upgrade transformers
```

## 生成语音
让我们从探索文本到语音的生成开始。首先，正如音频分类和自动语音识别一样，我们需要定义流水线。我们将定义一个文本到语音流水线，因为它能最好地描述我们的任务，并使用 [`suno/bark-small`](https://huggingface.co/suno/bark-small) 检查点：

```python
from transformers import pipeline

pipe = pipeline("text-to-speech", model="suno/bark-small")
```

下一步很简单，只需通过流水线传递一些文本。所有的预处理工作都将在流水线内完成：

```python
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
output = pipe(text)
```

在笔记本中，我们可以使用以下代码段来监听结果：

```python
from IPython.display import Audio

Audio(output["audio"], rate=output["sampling_rate"])
```

我们在流水线中使用的 Bark 模型实际上是多语言的，因此我们可以轻松地将初始文本替换为法语文本，并以完全相同的方式使用流水线。它会自己识别语言：

```python
fr_text = "Contrairement à une idée répandue, le nombre de points sur les élytres d'une coccinelle ne correspond pas à son âge, ni en nombre d'années, ni en nombre de mois. "
output = pipe(fr_text)
Audio(output["audio"], rate=output["sampling_rate"])
```

该模型不仅能使用多种语言，还能通过非语言交流和唱歌来生成音频。下面是如何让它唱歌的方法：

```python
song = "♪ In the jungle, the mighty jungle, the ladybug was seen. ♪ "
output = pipe(song)
Audio(output["audio"], rate=output["sampling_rate"])
```

我们将在后面的 "文本-语音 "单元中深入探讨 `Bark`的具体细节，并将展示如何使用其他模型来完成这项任务。现在，让我们来生成一些音乐！

## 生成音乐
和以前一样，我们将首先实例化一个流水线。对于音乐生成，我们将定义一个文本到音频的流水线，并使用预训练的检查点 [`facebook/musicgen-small`](https://huggingface.co/facebook/musicgen-small) 对其进行初始化。

```python
music_pipe = pipeline("text-to-audio", model="facebook/musicgen-small")
```

让我们为想要生成的音乐创建一段文字描述：

```python
text = "90s rock song with electric guitar and heavy drums"
```

我们可以通过向模型传递额外的 `max_new_tokens` 参数来控制生成输出的长度。

```python
forward_params = {"max_new_tokens": 512}

output = music_pipe(text, forward_params=forward_params)
Audio(output["audio"][0], rate=output["sampling_rate"])
```

