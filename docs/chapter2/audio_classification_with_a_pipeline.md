# 使用流水线进行音频分类
音频分类包括根据录音内容为其分配一个或多个标签。这些标签可以对应不同的声音类别，如音乐、语音或噪音，或者更具体的类别，如鸟鸣或汽车引擎声。

在详细了解最流行的音频Transformer的工作原理和微调自定义模型之前，让我们来看看如何利用Hugging Face `Transformers`，只需几行代码就能使用现成的预训练模型进行音频分类。

让我们继续使用与上一单元相同的 [MINDS-14 数据集](https://huggingface.co/datasets/PolyAI/minds14)。如果你还记得，MINDS-14 包含了人们用多种语言和方言向电子银行系统提问的录音，并且每段录音都有 `intent_class` 。我们可以根据通话的意图对录音进行分类。

和之前一样，我们先加载 `en-AU` 数据子集来试用流水线，并将其升频至 16kHz 采样率，这是大多数语音模型所要求的。

```python
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
```

要将录音归入一组类别，我们可以使用Hugging Face `Transformers` 中的音频分类流水线。在我们的案例中，我们需要一个针对意图分类进行过微调的模型，特别是在 MINDS-14 数据集上。幸运的是，Hub 中就有一个这样的模型！让我们使用 `pipeline() `函数加载它：

```python
from transformers import pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)
```

该流水线希望音频数据是一个 NumPy 数组。所有对原始音频数据的预处理都将由该流水线轻松处理。让我们选取一个例子来试试：

```python
example = minds[0]
```

如果你还记得数据集的结构，原始音频数据存储在`["audio"]["array"]`下的 NumPy 数组中，让我们直接将其传递给 `classifier`：

```python
classifier(example["audio"]["array"])
```

输出：

```python
[
    {"score": 0.9631525278091431, "label": "pay_bill"},
    {"score": 0.02819698303937912, "label": "freeze"},
    {"score": 0.0032787492964416742, "label": "card_issues"},
    {"score": 0.0019414445850998163, "label": "abroad"},
    {"score": 0.0008378693601116538, "label": "high_value_payment"},
]
```

该模型非常确信来电者打算了解有关支付账单的信息。让我们看看这个例子的实际标签是什么：

```python
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])
```

输出：

```python
"pay_bill"
```

太好了！预测的标签是正确的！在这里，我们很幸运地找到了一个能够准确分类我们所需的标签的模型。很多时候，在处理分类任务时，预训练模型的类别集与你需要模型区分的类别并不完全相同。在这种情况下，可以对预先训练好的模型进行微调，使其 "校准 "到准确的标签集。我们将在接下来的单元中学习如何做到这一点。现在，让我们来看看语音处理中另一项非常常见的任务--*自动语音识别*。