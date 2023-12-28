# 微调音乐分类模型
在本节中，我们将逐步介绍如何微调用于音乐分类的纯编码器Transformer模型。我们将使用一个轻量级模型进行演示，并使用相当小的数据集，这意味着代码可以在任何消费级 GPU（包括 Google Colab 免费提供的 T4 16GB GPU）上端对端运行。本节包括各种提示，如果您的 GPU 较小，在运行过程中遇到内存问题，可以尝试使用。

## 数据集
为了训练我们的模型，我们将使用 [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) 数据集，这是一个包含 1000 首歌曲的流行音乐流派分类数据集。每首歌都是 10 种音乐类型中的一种的 30 秒片段，从迪斯科到金属。我们可以使用 Hugging Face `Datasets` 中的 `load_dataset()` 函数从 Hugging Face Hub 获取音频文件及其相应的标签：

```python
from datasets import load_dataset

gtzan = load_dataset("marsyas/gtzan", "all")
gtzan
```

输出：

```python
Dataset({
    features: ['file', 'audio', 'genre'],
    num_rows: 999
})
```

> GTZAN 中的一个录音已损坏，因此已从数据集中删除。这就是为什么我们有 999 个示例而不是 1000 个示例的原因。

GTZAN 没有提供预定义的验证集，因此我们必须自己创建一个。数据集在不同类型之间是平衡的，因此我们可以使用 `train_test_split()` 方法快速创建一个 90/10 的分割集，如下所示：

```python
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
gtzan
```

输出：

```python
DatasetDict({
    train: Dataset({
        features: ['file', 'audio', 'genre'],
        num_rows: 899
    })
    test: Dataset({
        features: ['file', 'audio', 'genre'],
        num_rows: 100
    })
})
```

很好，现在我们已经有了训练集和验证集，让我们来看看其中一个音频文件：

```python
gtzan["train"][0]
```

输出：

```python
{
    "file": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
    "audio": {
        "path": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
        "array": array(
            [
                0.10720825,
                0.16122437,
                0.28585815,
                ...,
                -0.22924805,
                -0.20629883,
                -0.11334229,
            ],
            dtype=float32,
        ),
        "sampling_rate": 22050,
    },
    "genre": 7,
}
```

正如我们在[第一单元](chapter1/introduction_to_audio_data.md)中所看到的，音频文件以一维 NumPy 数组表示，数组的值代表该时间步的振幅。这些歌曲的采样率为 22,050 Hz，这意味着每秒采样 22,050 个振幅值。在使用不同采样率的预训练模型时，我们必须牢记这一点，自行转换采样率以确保两者匹配。我们还可以看到流派是以整数或类别标签的形式表示的，这也是模型进行预测的格式。让我们使用流派特征的 `int2str()` 方法将这些整数映射为人类可读的名称：

```python
id2label_fn = gtzan["train"].features["genre"].int2str
id2label_fn(gtzan["train"][0]["genre"])
```

输出：

```python
'pop'
```

这个标签看起来是正确的，因为它与音频文件的文件名一致。现在，让我们通过 Gradio 使用 Blocks API 创建一个简单的界面，再听几个例子：

```python
import gradio as gr


def generate_audio():
    example = gtzan["train"].shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label_fn(example["genre"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
```

从这些样本中，我们当然可以听出不同流派之间的差异，但Transformer也能做到这一点吗？让我们训练一个模型来找出答案！首先，我们需要为这项任务找到一个合适的预训练模型。让我们看看如何做到这一点。

## 为音频分类挑选预训练模型
首先，让我们为音频分类挑选一个合适的预训练模型。在这一领域，预训练通常在大量未标注的音频数据上进行，使用的数据集包括 [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) 和 [Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)。在Hugging Face Hub上找到这些模型的最佳方法是使用 "音频分类 "过滤器，如上一节所述。虽然像 Wav2Vec2 和 HuBERT 这样的模型非常流行，但我们将使用一种名为 *DistilHuBERT* 的模型。它是 [HuBERT](https://huggingface.co/docs/transformers/model_doc/hubert) 模型的缩小（或蒸馏）版本，训练速度提高了约 73%，但仍保留了大部分性能。

## 从音频到机器学习特征
## 预处理数据

与 NLP 中的tokenization类似，音频和语音模型需要将输入编码为模型可以处理的格式。在Hugging Face Transformers中，从音频到输入格式的转换由模型的*特征提取器*处理。与tokenizers类似，Hugging Face Transformers 提供了一个方便的 `AutoFeatureExtractor` 类，可以为给定的模型自动选择正确的特征提取器。为了了解我们如何处理音频文件，让我们从预先训练的检查点开始实例化 DistilHuBERT 的特征提取器：

```python
from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
```

由于模型和数据集的采样率不同，我们必须将音频文件重新采样为 16,000 Hz，然后再将其传递给特征提取器。为此，我们可以先从特征提取器中获取模型的采样率：

```python
sampling_rate = feature_extractor.sampling_rate
sampling_rate
```

输出：

```python
16000
```

接下来，我们使用 `cast_column() `方法和Hugging Face Datasets的音频特征对数据集进行重新采样：

```python
from datasets import Audio

gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

现在，我们可以检查数据集train-split的第一个样本，以验证其频率确实为 16,000 Hz。Hugging Face Datasets会在我们加载每个音频样本时对音频文件进行即时重采样：

```python
gtzan["train"][0]
```

输出：

```python
{
    "file": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
    "audio": {
        "path": "~/.cache/huggingface/datasets/downloads/extracted/fa06ce46130d3467683100aca945d6deafb642315765a784456e1d81c94715a8/genres/pop/pop.00098.wav",
        "array": array(
            [
                0.0873509,
                0.20183384,
                0.4790867,
                ...,
                -0.18743178,
                -0.23294401,
                -0.13517427,
            ],
            dtype=float32,
        ),
        "sampling_rate": 16000,
    },
    "genre": 7,
}
```

很好！我们可以看到，采样率已降至 16kHz。数组值也不同了，因为我们现在每 1.5 个振幅值中只有一个振幅值。

Wav2Vec2 和 HuBERT 类似模型的一个显著特点是，它们接受与语音信号原始波形相对应的浮点数组作为输入。这与 Whisper 等其他模型截然不同，在 Whisper 模型中，我们将原始音频波形预处理为频谱图格式。

我们提到过，音频数据是以一维数组的形式表示的，因此它已经具备了模型可以读取的正确格式（一组离散时间步长的连续输入）。那么，特征提取器究竟要做什么呢？

音频数据的格式是正确的，但我们对其取值没有施加任何限制。为了让模型达到最佳工作状态，我们希望所有输入都保持在相同的动态范围内。这将确保我们的样本获得相似范围的激活和梯度，有助于训练过程中的稳定性和收敛性。

为此，我们要对音频数据进行归一化处理，将每个样本的均值和单位方差重新缩放为零，这一过程称为*特征缩放*。我们的特征提取器执行的正是这种特征归一化！

我们可以将特征提取器应用于第一个音频样本，看看它的运行情况。首先，计算原始音频数据的均值和方差：

```python
import numpy as np

sample = gtzan["train"][0]["audio"]

print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
```

输出：

```python
Mean: 0.000185, Variance: 0.0493
```

我们可以看到，平均值已经接近零，但方差更接近 0.05。如果样本的方差更大，可能会给我们的模型带来问题，因为音频数据的动态范围会非常小，从而难以分离。让我们应用特征提取器，看看输出结果如何：

```python
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

print(f"inputs keys: {list(inputs.keys())}")

print(
    f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
)
```

输出：

```python
inputs keys: ['input_values', 'attention_mask']
Mean: -4.53e-09, Variance: 1.0
```

好吧！我们的特征提取器会返回一个包含两个数组的字典：input_values 和 attention_mask。input_values 是经过预处理的音频输入，我们会将其传递给 HuBERT 模型。[attention_mask](https://huggingface.co/docs/transformers/glossary#attention-mask) 用于一次性处理一批音频输入--它用来告诉模型我们在哪些地方填充了不同长度的输入。

我们可以看到，现在的平均值非常接近零，方差也接近一！这正是我们在将音频样本输入 HuBERT 模型之前所希望得到的结果。

> 请注意我们是如何将音频数据的采样率传递给特征提取器的。这是一种很好的做法，因为特征提取器会在内部进行检查，确保音频数据的采样率与模型预期的采样率一致。如果音频数据的采样率与模型的采样率不一致，我们就需要对音频数据进行上采样或下采样，使其达到正确的采样率。
>

很好，现在我们知道了如何处理重新采样的音频文件，最后要做的就是定义一个函数，将其应用于数据集中的所有示例。由于我们希望音频片段的长度为 30 秒，因此我们还将使用特征提取器的 `max_length` 和 `truncation` 参数截断更长的片段，如下所示：

```python
max_duration = 30.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs
```

定义好函数后，我们就可以使用 [`map() `](huggingface.co/docs/datasets/v2.14.0/en/package_reference/main_classes#datasets.Dataset.map)方法将其应用于数据集了。`.map()` 方法支持批量处理示例，我们将通过设置 `batched=True` 启用该方法。默认批次大小为 1000，但我们会将其减小到 100，以确保峰值 RAM 保持在 Google Colab 免费层级的合理范围内：

```python
gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
gtzan_encoded
```

输出：

```python
DatasetDict({
    train: Dataset({
        features: ['genre', 'input_values','attention_mask'],
        num_rows: 899
    })
    test: Dataset({
        features: ['genre', 'input_values','attention_mask'],
        num_rows: 100
    })
})
```

>  如果在执行上述代码时耗尽了设备的 RAM，可以调整批处理参数来降低 RAM 的峰值使用率。特别是可以修改以下两个参数：`batch_size`: 默认为 1000，但上面设置为 100。尝试将其减少 2 倍，即减少到 50。 `writer_batch_size`: 默认为 1000。尝试减小到 500，如果还不行，再减小 2 倍到 250。

为了简化训练，我们删除了数据集中的`audio`和`file`列。`input_values` 列包含编码后的音频文件，attention_mask 是由 0/1 值组成的二进制掩码，表示我们对音频输入进行了填充，而 `genre` 列则包含相应的标签（或目标）。为了让训练器能够处理类别标签，我们需要将流派列重命名为标签：

```python
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
```

最后，我们需要从数据集中获取标签映射。这种映射将把我们从整数 id（如 7）转换为人类可读的类别标签（如 `"pop"`），然后再返回。这样，我们就可以将模型的整数 id 预测转换为人类可读的格式，使我们能够在任何下游应用程序中使用该模型。我们可以通过使用 `int2str()` 方法来实现这一目的，具体如下：

```python
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

id2label["7"]
```

```python
'pop'
```

## 微调模型
为了微调模型，我们将使用 Hugging Face Transformers 中的`Trainer`类。正如我们在其他章节中所看到的，`Trainer`是一个高级 API，旨在处理最常见的训练场景。在本例中，我们将使用`Trainer`来微调 GTZAN 上的模型。为此，我们首先需要为该任务加载一个模型。我们可以使用 `AutoModelForAudioClassification` 类来实现这一点，该类将自动为我们预训练的 DistilHuBERT 模型添加适当的分类头。让我们继续实例化模型：

```python
from transformers import AutoModelForAudioClassification

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
```

我们强烈建议您在训练时直接将模型检查点上传到 [Hugging Face Hub](https://huggingface.co/)。Hub 提供

+ 集成版本控制：确保训练过程中不会丢失任何模型检查点。
+ Tensorboard 日志：跟踪训练过程中的重要指标。
+ 模型卡片：记录模型的作用及其预期用例。
+ 社区：与社区共享和协作的简便方法！

将笔记本链接到 Hub 非常简单，只需在出现提示时输入您的 Hub 身份验证令牌即可。[在此查找](https://huggingface.co/settings/tokens)您的 Hub 身份验证令牌：

```python
from huggingface_hub import notebook_login

notebook_login()
```

输出：

```python
Login successful
Your token has been saved to /root/.huggingface/token
```

下一步是定义训练参数，包括批量大小、梯度累积步骤、epoch数和学习率：

```python
from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
)
```

> 这里我们设置了 `push_to_hub=True`，以便在训练过程中自动上传微调后的检查点。如果不希望将检查点上传到Hub，可以将其设置为 `False`。

我们需要做的最后一件事是定义指标。由于数据集是平衡的，我们将使用准确性作为衡量标准，并使用 Hugging Face `Evaluate` 库加载它：

```python
import evaluate
import numpy as np

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

现在，我们已经准备好了一切！让我们实例化训练器并训练模型：

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
```

> 根据 GPU 的不同，在开始训练时可能会遇到 CUDA "内存不足 "的错误。在这种情况下，可以将批处理量（`batch_size`）按 2 倍递增，并使用梯度累积步骤（`gradient_accumulation_steps`）进行补偿。

输出：

```python
| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.7297        | 1.0   | 113  | 1.8011          | 0.44     |
| 1.24          | 2.0   | 226  | 1.3045          | 0.64     |
| 0.9805        | 3.0   | 339  | 0.9888          | 0.7      |
| 0.6853        | 4.0   | 452  | 0.7508          | 0.79     |
| 0.4502        | 5.0   | 565  | 0.6224          | 0.81     |
| 0.3015        | 6.0   | 678  | 0.5411          | 0.83     |
| 0.2244        | 7.0   | 791  | 0.6293          | 0.78     |
| 0.3108        | 8.0   | 904  | 0.5857          | 0.81     |
| 0.1644        | 9.0   | 1017 | 0.5355          | 0.83     |
| 0.1198        | 10.0  | 1130 | 0.5716          | 0.82     |
```

根据您的 GPU 或分配给 Google Colab 的 GPU，训练大约需要 1 个小时。我们的最佳评估准确率为 83%--对于仅有 10 个epoch和 899 个示例的训练数据来说还不错！当然，我们还可以通过训练更多的epoch、使用正则化技术（如 dropout）或将每个音频示例从 30 秒细分为 15 秒，从而使用更高效的数据预处理策略来改进这一结果。

目前最大的问题是，与其他音乐分类系统相比，我们的模型有何优势？为此，我们可以查看自动评估排行榜（[autoevaluate leaderboard](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=marsyas%2Fgtzan&only_verified=0&task=audio-classification&config=all&split=train&metric=accuracy))，该排行榜按语言和数据集对模型进行分类，然后根据准确率对它们进行排名。

当我们将训练结果推送到 Hub 时，我们可以自动向排行榜提交我们的检查点--我们只需设置适当的关键字参数（kwargs）即可。您可以根据自己的数据集、语言和模型名称更改这些值：

```python
kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}
```

现在可以将训练结果上传到 Hub。为此，请执行 `.push_too_hub` 命令：

```python
trainer.push_too_hub(**kwargs)
```

这将把训练日志和模型权重保存在 `"your-username/distilhubert-finetuned-gtzan "`下。本例中，请查看 [`"sanchit-gandhi/distilhubert-finetuned-gtzan"`](https://huggingface.co/sanchit-gandhi/distilhubert-finetuned-gtzan)下的上传。

## 共享模型
现在，您可以使用 Hub 上的链接与任何人共享此模型。他们可以使用标识符 `"your-username/distilhubert-finetuned-gtzan"`将其直接加载到 `pipeline()` 类中。例如，加载微调检查点 `"sanchit-gandhi/distilhubert-finetuned-gtzan"`：

```python
from transformers import pipeline

pipe = pipeline(
    "audio-classification", model="sanchit-gandhi/distilhubert-finetuned-gtzan"
)
```

## 结论
在本节中，我们逐步介绍了如何微调 DistilHuBERT 模型以进行音乐分类。虽然我们的重点是音乐分类任务和 GTZAN 数据集，但这里介绍的步骤更普遍地适用于任何音频分类任务--同样的脚本可用于口语音频分类任务，如关键词定位或语言识别。您只需将数据集换成与您感兴趣的任务相对应的数据集即可！如果您对微调其他用于音频分类的 Hugging Face Hub 模型感兴趣，我们建议您查看 Hugging Face Transformers repository中的其他[示例](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification)。

在下一节中，我们将使用你刚刚微调过的模型，创建一个音乐分类演示，你可以在 Hugging Face Hub 上分享。