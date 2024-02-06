# 语音识别评估和指标

如果您熟悉 NLP 中的[莱文斯坦距离](https://zh.wikipedia.org/wiki/%E8%90%8A%E6%96%87%E6%96%AF%E5%9D%A6%E8%B7%9D%E9%9B%A2)，那么评估语音识别系统的指标就不会陌生！如果您不熟悉，也不用担心，我们将从头到尾进行讲解，确保您了解不同的指标并理解它们的含义。

在评估语音识别系统时，我们会将系统的预测与目标文本转录进行比较，并标注存在的任何错误。我们将这些错误分为三类：

1. 替换 (S)：我们在预测中转录了**错误的单词**（"sit "而不是 "sat"）。
2. 插入 (I)：我们在预测中添加了一个**额外的单词**
3. 删除 (D)：在预测中**删除**一个单词

这些误差类别对所有语音识别指标都是一样的。不同的是我们计算这些误差的级别：我们可以在*单词级别*或*字符级别*计算误差。

我们将为每个指标定义使用一个运行示例。这里，我们有一个*基本事实*或*参考*文本序列：

```python
reference = "the cat sat on the mat"
```

以及我们正在评估的语音识别系统的预测序列：

```python
prediction = "the cat sit on the"
```

我们可以看到，预测结果非常接近，但有些单词并不完全正确。我们将根据三个最常用的语音识别指标的参考值来评估这一预测结果，看看我们在每个指标上得到了什么样的数据。

## 词错误率
*词错误率 (WER)* 指标是语音识别的 "事实 "指标。它以*单词*为单位计算替换、插入和删除。这意味着错误是逐字注释的。以我们的例子为例：

| Reference:  | the  | cat  | sat     | on   | the  | mat  |
| ----------- | ---- | ---- | ------- | ---- | ---- | ---- |
| Prediction: | the  | cat  | **sit** | on   | the  |      |
| Label:      | ✅    | ✅    | S       | ✅    | ✅    | D    |

在这里，我们有

+ 1 个替换（"sit "而不是 "sat"）
+ 0 个插入
+ 1 个删除（"mat "缺失）

总共有 2 个错误。为了得出错误率，我们将错误数除以参考文献中的单词总数（N），在本例中，参考文献中的单词总数为 6：
$$
\begin{aligned}
W E R & =\frac{S+I+D}{N} \\
& =\frac{1+0+1}{6} \\
& =0.333
\end{aligned}
$$
好吧！因此，我们的错误率为 0.333，即 33.3%。请注意，"sit "这个单词只有一个字符是错误的，但整个单词都被标记为错误。这是 WER 的一个显著特点：拼写错误无论多么轻微，都会受到重罚。

WER 的定义是*越低越好*：WER 越低意味着我们的预测错误越少，因此一个完美的语音识别系统的 WER 应该是零（没有错误）。

让我们看看如何使用 Hugging Face `Evaluate` 计算 WER。我们需要两个软件包来计算 WER 指标：Hugging Face `Evaluate` 用于 API 接口，而 `JIWER` 则负责运行计算的繁重工作：

```python
pip install --upgrade evaluate jiwer
```

太好了！现在我们可以加载 WER 指标，并计算出我们示例中的数字：

```python
from evaluate import load

wer_metric = load("wer")

wer = wer_metric.compute(references=[reference], predictions=[prediction])

print(wer)
```

打印输出：

```python
0.3333333333333333
```

0.33，即 33.3%，符合预期！我们现在知道了 WER 计算的原理。

现在，有一点让人很困惑......你认为 WER 的上限是多少？你会认为是 1 或 100% 对吗？不是的！因为 WER 是错误数与字数 (N) 的比率，所以 WER 没有上限！举个例子，我们预测了 10 个词，而目标词只有 2 个。如果我们的预测全部错误（10 个错误），那么我们的 WER 就是 10 / 2 = 5，即 500%！如果您在训练 ASR 系统时发现 WER 超过了 100%，那么这一点一定要牢记。不过，如果您看到的是这样的结果，那么很可能是出了什么问题...... 😅。

## 单词准确率

我们可以将 WER 反过来，给出一个越高越好的指标。我们可以测量系统的*单词准确率 (WAcc)*，而不是单词错误率：
$$
W A c c=1-W E R
$$


WAcc 也是在单词级别上测量的，只是将 WER 重新表述为准确度指标，而不是错误指标。WAcc 在语音文献中很少被引用--我们认为系统预测的是单词错误，因此更倾向于使用与这些错误类型注释更相关的错误率指标。

## 字符错误率

我们将 "sit "的整个单词标注为错误，而实际上只有一个字母是错误的，这似乎有点不公平。这是因为我们是在单词层面上评估我们的系统，从而逐词标注错误。*字符错误率 (CER)* 是在*字符*层面对系统进行评估。这意味着我们将单词划分为单个字符，并逐个字符标注错误：

| Reference:  | t    | h    | e    |      | c    | a    | t    |      | s    | a     | t    |      | o    | n    |      | t    | h    | e    |      | m    | a    | t    |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Prediction: | t    | h    | e    |      | c    | a    | t    |      | s    | **i** | t    |      | o    | n    |      | t    | h    | e    |      |      |      |      |
| Label:      | ✅    | ✅    | ✅    |      | ✅    | ✅    | ✅    |      | ✅    | S     | ✅    |      | ✅    | ✅    |      | ✅    | ✅    | ✅    |      | D    | D    | D    |

现在我们可以看到，在 "sit "这个单词中，"s "和 "t "被标记为正确。只有 "i "被标记为替换错误 (S)。因此，我们会奖励我们的系统部分正确的预测🤝。

在我们的示例中，有 1 个字符被替换，0 个字符被插入，3 个字符被删除。总共有 14 个字符。因此，我们的 CER 是：

没错！我们的 CER 为 0.286，即 28.6%。请注意，这比我们的 WER 要低--我们对拼写错误的惩罚要少得多。
$$
\begin{aligned}
C E R & =\frac{S+I+D}{N} \\
& =\frac{1+0+3}{14} \\
& =0.286
\end{aligned}
$$




## 我应该使用哪个指标？
一般来说，在评估语音系统时，WER 比 CER 更常用。这是因为 WER 要求系统对预测的上下文有更深入的理解。在我们的例子中，"sit "的时态是错误的。如果系统能够理解动词与句子时态之间的关系，就会预测出正确的动词时态 "sat"。我们希望我们的语音系统能达到这种理解水平。因此，尽管 WER 没有 CER 那么宽容，但它也更有利于我们想要开发的可理解系统。因此，我们通常使用 WER，并鼓励您也这样做！但是，在某些情况下，我们无法使用 WER。某些语言，如普通话和日语，没有 "词"的概念，因此 WER 没有意义。在这里，我们恢复使用 CER。

在我们的例子中，计算 WER 时只使用了一个句子。在评估真实系统时，我们通常会使用由数千个句子组成的整个测试集。在对多个句子进行评估时，我们会在所有句子中汇总 S、I、D 和 N，然后根据上述公式计算 WER。这样可以更好地估计未见数据的 WER。

## 标准化

如果我们在带有标点符号和大小写的数据上训练 ASR 模型，它将学会在转录中预测大小写和标点符号。当我们想将模型用于实际的语音识别应用（如转录会议内容或口述内容）时，这就非常好，因为预测的转录内容将完全符合大小写和标点符号的格式，这种风格被称为*正字法*。

不过，我们也可以选择对数据集进行标准化处理，以去除任何大小写和标点符号。对数据集进行标准化处理可使语音识别任务变得更容易：模型不再需要区分大小写字符，也不必仅从音频数据中预测标点符号（例如，分号发出什么声音？） 正因为如此，单词错误率自然会降低（这意味着结果会更好）。Whisper 论文证明了规范化转录对 WER 结果的巨大影响（参见 [Whisper](https://cdn.openai.com/papers/whisper.pdf) 论文的第 4.4 节）。虽然我们得到了较低的 WER，但该模型并不一定更适合生产。由于缺少大小写和标点符号，模型预测的文本明显难以阅读。以[上一节](chapter5/asr_models)中的例子为例(，我们在 LibriSpeech 数据集中的同一音频样本上运行了 Wav2Vec2 和 Whisper。Wav2Vec2 模型既不能预测标点符号，也不能预测大小写，而 Whisper 则能预测两者。并排比较这两种转录，我们发现 Whisper 的转录更容易阅读：

```python
Wav2Vec2:  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAUS AND ROSE BEEF LOOMING BEFORE US SIMALYIS DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
Whisper:   He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind.
```

Whisper 转录是正字法转录，因此可以随时使用--它的格式符合我们对会议转录或听写稿的预期，并带有标点符号和大小写。相反，如果我们想将 Wav2Vec2 预测结果用于下游应用，就需要使用额外的后处理来恢复标点和大小写。

在标准化和不标准化之间有一个很好的平衡点：我们可以在正字法转录上训练我们的系统，然后在计算 WER 之前将预测和目标标准化。这样，我们既能训练系统预测完全格式化的文本，又能受益于转录标准化带来的 WER 改进。

Whisper 模型带有一个标准化器，可以有效处理大小写、标点符号和数字格式等规范化问题。让我们将标准化器应用到 Whisper 转录中，演示如何将其标准化：

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

prediction = " He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly is drawn from eating and its results occur most readily to the mind."
normalized_prediction = normalizer(prediction)

normalized_prediction
```

输出：

```python
' he tells us that at this festive season of the year with christmas and roast beef looming before us similarly is drawn from eating and its results occur most readily to the mind '
```

好极了！我们可以看到，文本已完全小写，所有标点符号都已去除。现在我们来定义参考转录，然后计算参考和预测之间的标准化 WER：

```python
reference = "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND"
normalized_referece = normalizer(reference)

wer = wer_metric.compute(
    references=[normalized_referece], predictions=[normalized_prediction]
)
wer
```

输出：

```python
0.0625
```

6.25% - 这与我们对 LibriSpeech 验证集上 Whisper 基本模型的预期差不多。正如我们在这里看到的，我们预测的是正字法转录，但在计算 WER 之前对参考和预测进行了标准化处理，从而提高了 WER。

如何标准化转录最终取决于您的需求。我们建议在正字法文本上进行训练，并在规范化文本上进行评估，以获得两全其美的效果。

## 把所有东西放在一起
好了！本单元到目前为止，我们已经介绍了三个主题：预训练模型、数据集选择和评估。我们将在 Common Voice 13 Dhivehi 测试集上评估预训练的 Whisper 模型，为下一节的微调做好准备。我们将把得到的 WER 数值作为微调运行的基线，或者说是我们要尝试击败的目标数值🥊。

首先，我们将使用 `pipeline()` 类加载预训练的 Whisper 模型。这个过程现在已经非常熟悉了！我们唯一要做的新事情是，如果在 GPU 上运行，则以半精度（float16）加载模型--这将加快推理速度，而对 WER 精度几乎没有影响。

```python
from transformers import pipeline
import torch

if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    torch_dtype=torch_dtype,
    device=device,
)
```

接下来，我们将加载 Common Voice 13 的 Dhivehi 测试集。大家应该还记得，在上一节中，Common Voice 13 是有*限制*的，这意味着我们在访问数据集之前必须同意数据集的使用条款。现在，我们可以将 Hugging Face 账户链接到笔记本上，这样就可以通过当前使用的机器访问数据集了。

将笔记本链接到 Hub 非常简单，只需根据提示输入 Hub 验证令牌即可。在[这里](https://huggingface.co/settings/tokens)找到你的 Hub 身份验证令牌，按提示输入即可：

```python
from huggingface_hub import notebook_login

notebook_login()
```

太好了！将笔记本链接到 Hugging Face 账户后，我们就可以继续下载通用语音数据集了。下载和预处理需要几分钟时间，从 Hugging Face Hub 获取数据并自动在笔记本上进行准备：

```python
from datasets import load_dataset

common_voice_test = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)
```

<blockquote style="background-color: #e0f2fe; padding: 10px; border-left: 5px solid #5c9ecf;">
如果您在加载数据集时遇到身份验证问题，请确保您已通过以下链接接受了Hugging Face Hub上的数据集使用条款： https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0
</blockquote>

对整个数据集进行评估的方法与对单个示例进行评估的方法大致相同--我们要做的就是**循环**播放输入的音频，而不是只推断单个样本。为此，我们首先要将数据集转换为 KeyDataset。这样做的目的只是挑出我们要转发给模型的特定数据集列（在我们的例子中，就是 `"audio"`列），忽略其他列（比如target transcriptions，我们不想用它来推断）。然后，我们遍历转换后的数据集，将模型输出附加到列表中以保存预测结果。如果在半精度 GPU 上运行，以下代码单元将耗时约 5 分钟，峰值内存为 12GB：

```python
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

all_predictions = []

# 进行流式化推理
for prediction in tqdm(
    pipe(
        KeyDataset(common_voice_test, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(common_voice_test),
):
    all_predictions.append(prediction["text"])
```

<blockquote style="background-color: #e0f2fe; padding: 10px; border-left: 5px solid #5c9ecf;">
如果在运行上述单元时遇到 CUDA 内存不足 (OOM)，请将 "batch_size "按 2 倍递减，直到找到适合您设备的批次大小。
</blockquote>

最后，我们可以计算 WER。首先计算正交 WER，即不经过任何后处理的 WER：

```python
from evaluate import load

wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=common_voice_test["sentence"], predictions=all_predictions
)
wer_ortho
```

输出：

```python
167.29577268612022
```

好吧......167% 基本上意味着我们的模型输出的是垃圾😜 不用担心，我们的目标是通过在 Dhivehi 训练集上对模型进行微调来改善这一结果！

接下来，我们将评估归一化 WER，即经过归一化后处理的 WER。我们必须过滤掉归一化后为空的样本，否则参考（N）中的单词总数将为零，这将导致我们的计算出现除以零的误差：

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()

# 计算标准化WER
all_predictions_norm = [normalizer(pred) for pred in all_predictions]
all_references_norm = [normalizer(label) for label in common_voice_test["sentence"]]

# 过滤步骤，只评估与非零引用相对应的样本
all_predictions_norm = [
    all_predictions_norm[i]
    for i in range(len(all_predictions_norm))
    if len(all_references_norm[i]) > 0
]
all_references_norm = [
    all_references_norm[i]
    for i in range(len(all_references_norm))
    if len(all_references_norm[i]) > 0
]

wer = 100 * wer_metric.compute(
    references=all_references_norm, predictions=all_predictions_norm
)

wer
```

输出：

```python
125.69809089960707
```

我们再次看到，通过对参照和预测进行标准化处理，我们的误码率大幅降低：基线模型的正字法测试误码率为 168%，而标准化后的误码率为 126%。

没错！当我们对模型进行微调时，我们希望能超越这些数字，从而改进用于迪维希语语音识别的 Whisper 模型。继续阅读，亲身体验微调示例 🚀