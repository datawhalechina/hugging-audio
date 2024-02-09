# Fine-tuning the ASR model

在本节中，我们将逐步介绍如何微调 Whisper，以便在 Common Voice 13 数据集上进行语音识别。我们将使用模型的 "小 "版本和相对轻量级的数据集，使您能够在任何磁盘空间要求较低的 16GB 以上 GPU（如 Google Colab 免费提供的 16GB T4 GPU）上快速运行微调。

如果您的 GPU 较小，或者在训练过程中遇到内存问题，可以根据提供的建议减少内存使用量。反之，如果您的 GPU 较大，则可以修改训练参数，最大限度地提高吞吐率。因此，无论您的 GPU 规格如何，都可以使用本指南！

同样，本指南概述了如何针对迪维希语微调 Whisper 模型。不过，这里涵盖的步骤可通用于 Common Voice 数据集中的任何语言，更广泛地说，可通用于 Hugging Face Hub 上的任何 ASR 数据集。您可以调整代码，快速切换到您选择的语言，并用您的母语微调 Whisper 模型🌍。

好的！现在，让我们开始启动微调流水线！

## 准备环境
我们强烈建议您在训练时直接将模型检查点上传到 [Hugging Face Hub](https://huggingface.co/)。Hub 可提供

+ 集成版本控制：确保训练过程中不会丢失任何模型检查点。
+ Tensorboard 日志：跟踪训练过程中的重要指标。
+ 模型卡片：记录模型的作用及其预期用例。
+ 社区：与社区共享和协作的简便方法！🤗

将笔记本链接到 Hub 非常简单，只需在出现提示时输入您的 Hub 身份验证令牌即可。[在此](https://huggingface.co/settings/tokens)查找您的 Hub 身份验证令牌，并按提示输入：

```python
from huggingface_hub import notebook_login

notebook_login()
```

输出：

```python
Login successful
Your token has been saved to /root/.huggingface/token
```

## 加载数据集
[Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) 包含大约 10 个小时的迪维希语标签数据，其中 3 个小时是保留的测试数据。用于微调的数据极少，因此我们将依靠 Whisper 在低资源迪维希语预训练期间获得的大量多语言 ASR 知识。

使用 🤗 `Datasets`，下载和准备数据非常简单。只需一行代码，我们就能下载和准备 Common Voice 13 分割数据。由于迪维希语的资源非常少，我们将把`训练`和`验证`集结合起来，以获得大约 7 个小时的训练数据。我们将使用三小时的测试数据作为我们的保留`测试`集：

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)

print(common_voice)
```

输出：

```python
DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 4904
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant'],
        num_rows: 2212
    })
})
```

> 您可以将语言标识符从`"dv"`更改为自己选择的语言标识符。要查看 Common Voice 13 中所有可能的语言，请访问 Hugging Face Hub 的数据集卡：https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0。

大多数 ASR 数据集只提供输入音频样本（`音频`）和相应的转录文本（`句子`）。普通语音包含额外的元数据信息，如口音和地域，我们可以忽略这些信息用于 ASR。为了尽可能保持笔记本的通用性，我们只考虑对输入音频和转录文本进行微调，而忽略了额外的元数据信息：

```python
common_voice = common_voice.select_columns(["audio", "sentence"])
```

## 特征提取器、分词器和处理器

ASR 流水线可分为三个阶段：

1. 特征提取器，将原始音频输入预处理为对数梅尔频谱图
2. 执行序列到序列映射的模型
3. 分词器，将预测的 token 处理为文本

在 🤗 Transformers 中，Whisper 模型有一个相关的特征提取器和分词器，分别称为 [`WhisperFeatureExtractor`](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperFeatureExtractor) 和 [`WhisperTokenizer`](https://huggingface.co/docs/transformers/main/model_doc/whisper#transformers.WhisperTokenizer)。为了简化我们的工作，这两个对象被封装在一个名为 `WhisperProcessor` 的类中。我们可以调用 [`WhisperProcessor`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor) 来执行音频预处理和文本标记后处理。这样，我们在训练过程中只需跟踪两个对象：处理器和模型。

在进行多语言微调时，我们需要在处理器实例化时设置 `"语言"` 和 `"任务"` 。`"语言"` 应设置为源音频语言，任务应设置为 `"转录"`（用于语音识别）或 `"翻译"`（用于语音翻译）。这些参数会修改标记化器的行为，应正确设置以确保目标标签编码正确。

我们可以通过导入语言列表查看 Whisper 支持的所有可能语言：

```python
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

TO_LANGUAGE_CODE
```

如果您滚动浏览这个列表，您会发现其中包含了许多语言，但迪维希语是少数几种没有包含的语言之一！这意味着 Whisper 没有在迪维希语上进行预先训练。但是，这并不意味着我们不能对 Whisper 进行微调。这样做，我们将教 Whisper 一门新语言，一门预先训练的检查点不支持的语言。这很酷，对吧！

当你在一种新语言上进行微调时，Whisper 会很好地利用它预先训练过的其他 96 种语言的知识。大体上说，所有现代语言都与 Whisper 已掌握的 96 种语言中的至少一种在语言上相似，因此我们将采用这种跨语言知识表示范例。

要在一种新语言上对 Whisper 进行微调，我们需要做的是找到与 Whisper 预先训练过的语言**最相似**的语言。维基百科上关于迪维希语的文章指出，迪维希语与斯里兰卡的僧伽罗语密切相关。如果我们再次检查语言代码，就会发现僧伽罗语存在于 Whisper 语言集中，因此我们可以放心地将语言参数设置为 `"僧伽罗语"`。

好的！我们将从预先训练好的检查点加载处理器，将语言设置为 `"僧伽罗语"`，并如上所述将任务设置为 `"转录"`：

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="sinhalese", task="transcribe"
)
```

值得重申的是，在大多数情况下，您会发现要微调的语言在预训练语言集中，在这种情况下，您只需将该语言直接设置为源音频语言即可！请注意，如果只进行英语微调，这两个参数都应省略，因为在英语微调中，语言（"English"）和任务（"transcribe"）只有一个选项。

## 预处理数据

让我们来看看数据集的特征。请特别注意 `"audio"`一栏，它详细说明了音频输入的采样率：

```python
common_voice["train"].features
```

输出：

```python
{'audio': Audio(sampling_rate=48000, mono=True, decode=True, id=None),
 'sentence': Value(dtype='string', id=None)}
```

由于输入音频的采样率为 48kHz，因此在将其传递给 Whisper 特征提取器之前，我们需要将其采样率*降低*为 16kHz，16kHz 是 Whisper 模型所期望的采样率。

我们将使用 `datasets` 的 `cast_column` 方法将音频输入设置为正确的采样率。此操作不会就地更改音频，而是向数据集发出信号，以便在加载音频样本时对其进行即时重新采样：

```python
from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

现在我们可以编写一个函数，为模型准备数据：

1. 通过调用 `sample["audio"]`，我们将逐个样本加载并重新采样音频数据。如上所述，Hugging Face `Datasets` 会即时执行任何必要的重采样操作。
2. 我们使用特征提取器从一维音频数组中计算对数梅尔频谱输入特征。
3. 我们使用分词器将转录编码为标签 id。

```python
def prepare_dataset(example):
    audio = example["audio"]

    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example
```

我们可以使用 Hugging Face `Datasets` 的 `.map` 方法将数据准备函数应用于所有训练示例。我们将删除原始训练数据（音频和文本）中的列，只留下 `prepare_dataset` 函数返回的列：

```python
common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)
```

最后，我们会过滤任何音频样本长度超过 30 秒的训练数据。否则，这些样本会被 Whisper 特征提取器截断，从而影响训练的稳定性。我们定义了一个函数，该函数对小于 30 秒的样本返回 `True`，对超过 30 秒的样本返回 `False`：

```python
max_input_length = 30.0


def is_audio_in_length_range(length):
    return length < max_input_length
```

我们通过 Hugging Face `Datasets` 的 `.filter` 方法，将过滤函数应用于训练数据集的所有样本：

```python
common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
```

让我们来看看通过这一过滤步骤，我们删除了多少训练数据：

```python
common_voice["train"]
```

输出：

```python
Dataset({
    features: ['input_features', 'labels', 'input_length'],
    num_rows: 4904
})
```

好吧！在这种情况下，我们的样本数量与之前相同，因此没有超过 30 秒的样本。如果切换语言，情况可能就不是这样了，所以为了稳健起见，最好在原处保留这一过滤步骤。这样，我们就为训练做好了充分的数据准备！让我们继续看看如何使用这些数据对 Whisper 进行微调。

## 训练和评估
既然我们已经准备好了数据，那么就可以进入训练流程了。[🤗 Trainer](https://huggingface.co/docs/transformers/main/main_classes/trainer) 将为我们完成大部分繁重的工作。我们要做的就是

+ 定义数据整理器：数据整理器接收我们预处理过的数据，并为模型准备好 PyTorch 张量。
+ 评估指标：在评估过程中，我们希望使用字错误率（WER）指标来评估模型。我们需要定义一个 `compute_metrics` 函数来处理该计算。
+ 加载预训练的检查点：我们需要加载预训练的检查点，并为训练正确配置。
+ 定义训练参数：这些参数将被🤗 `Trainer` 用于构建训练计划。

对模型进行微调后，我们将在测试数据上对其进行评估，以验证我们是否已正确地训练它转录迪维希语语音。

### 定义数据整理器

序列到序列语音模型的数据整理器是独一无二的，因为它能独立处理 `input_features` 和 `labels` ：`input_features` 必须由特征提取器处理，`lablels` 则由分词器处理。

`input_features` 已经填充为 30s，并转换为固定维度的对数梅尔频谱图，因此我们只需将其转换为成批的 `PyTorch` 张量。我们使用特征提取器的 `.pad` 方法（`return_tensors=pt`）进行转换。请注意，由于输入的维度是固定的，因此这里没有应用额外的 padding ，输入特征只是被转换成 PyTorch 张量。

另一方面，`labels` 没有 padding。我们首先使用 tokenizer 的 `.pad` 方法将序列填充到批次中的最大长度。然后用 -100 替换填充标记，这样在计算损失时就**不会**考虑这些标记。然后，我们会从标签序列的开头剪切掉转录标记的开头，因为我们会在以后的训练中添加它。

我们可以利用之前定义的 `WhisperProcessor` 来执行特征提取和 tokenizer 操作：

```python
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```

现在我们可以初始化刚刚定义的数据整理器：

```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```

前进！

### 评价指标
接下来，我们要定义在评估集上使用的评估指标。我们将使用["评估"](chapter5/evaluation_and_metrics_for_speech_recognition.md)一节中介绍的词错误率 (WER) 指标，这是评估 ASR 系统的 "事实 "指标。

我们将从 🤗 `Evaluate` 中加载 WER 指标：

```python
import evaluate

metric = evaluate.load("wer")
```

然后，我们只需定义一个函数，接收我们的模型预测并返回 WER 指标。这个名为 `compute_metrics` 的函数首先会用 `label_ids` 中的 `pad_token_id` 替换 -100（取消我们在数据整理器中应用的步骤，以在损失中正确忽略填充标记）。然后将预测和 label ids 解码为字符串。最后，计算预测值和参考标签之间的 WER。在这里，我们可以选择使用 "标准化"转录和预测进行评估，因为 "标准化 "转录和预测去除了标点符号和大小写。我们建议您这样做，以受益于转录标准化带来的 WER 改善。

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}
```

### 加载预先训练好的检查点
现在，让我们加载预先训练好的 Whisper 小型检查点。同样，通过使用 🤗 `Transformers`，这也是微不足道的！

```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
```

我们将把训练时的 `use_cache` 设置为 `False`，因为我们使用的是[梯度检查点技术](https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing)，而这两者是不兼容的。我们还将覆盖两个生成参数，以控制模型在推理过程中的行为：我们将通过设置`language`和`task`参数，在生成过程中强制使用语言和任务token，并在生成过程中重新启用缓存，以加快推理时间：

```python
from functools import partial

# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, language="sinhalese", task="transcribe", use_cache=True
)
```

## 定义训练配置

最后一步，我们要定义与训练相关的所有参数。在这里，我们将训练步数设置为 500 步。与预训练的 Whisper 模型相比，这样的训练步数足以使 WER 有较大的提高，同时还能确保在 Google Colab 免费层上运行约 45 分钟即可完成微调。有关训练参数的更多详情，请参阅 [Seq2SeqTrainingArguments 文档](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments)。

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
```

> 如果不想将模型检查点上传到 Hub，请设置 `push_to_hub=False`。

我们可以将训练参数连同模型、数据集、数据整理器和 `compute_metrics` 函数一起转发给 🤗 `Trainer`：

这样，我们就可以开始训练了！

### 训练
要启动训练，只需执行即可：

```python
trainer.train()
```

训练大约需要 45 分钟，这取决于您的 GPU 或分配给 Google Colab 的 GPU。根据 GPU 的情况，开始训练时可能会遇到 CUDA "内存不足 "的错误。在这种情况下，可以将每台设备的训练批量（`per_device_train_batch_size`）减少 2 倍，并使用[梯度累积步骤](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_accumulation_steps)（`gradient_accumulation_steps`）进行补偿。

输出：

| Training Loss | Epoch | Step | Validation Loss | Wer Ortho | Wer     |
| ------------- | ----- | ---- | --------------- | --------- | ------- |
| 0.136         | 1.63  | 500  | 0.1727          | 63.8972   | 14.0661 |

我们的最终 WER 为 14.1%--对于 7 个小时的训练数据和仅 500 个训练步数来说，这已经很不错了！这相当于比预先训练的模型提高了 112%！这意味着，我们将一个之前对迪维希语一无所知的模型进行了微调，在不到一个小时的时间内就能准确识别迪维希语语音 🤯。

最大的问题是，它与其他 ASR 系统相比如何。为此，我们可以查看自动评估[排行榜](https://huggingface.co/spaces/autoevaluate/leaderboards?dataset=mozilla-foundation%2Fcommon_voice_13_0&only_verified=0&task=automatic-speech-recognition&config=dv&split=test&metric=wer)（autoevaluate leaderboard），该排行榜按语言和数据集对模型进行分类，然后根据它们的 WER 进行排名。

从排行榜上我们可以看到，我们训练了 500 步的模型令人信服地击败了我们在上一节中评估过的预训练 Whisper Small 检查点。干得漂亮 👏

我们看到有几个检查点比我们训练的检查点做得更好。Hugging Face Hub 的优点在于它是一个*协作*平台--如果我们自己没有时间或资源进行较长时间的训练，我们可以加载社区中其他人训练过并好心分享的检查点（确保为此感谢他们！）。使用 `pipeline` 类加载这些检查点的方式与我们之前使用预训练检查点的方式完全相同！因此，您完全可以在排行榜上挑选最佳模型用于您的任务！

当我们将训练结果推送到 Hub 时，我们可以自动向排行榜提交我们的检查点--我们只需设置适当的关键字参数（kwargs）即可。您可以根据自己的数据集、语言和模型名称更改这些值：

```python
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_13_0",
    "dataset": "Common Voice 13",  # a 'pretty' name for the training dataset
    "language": "dv",
    "model_name": "Whisper Small Dv - Sanchit Gandhi",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}
```

现在可以将训练结果上传到 Hub。为此，请执行 `push_to_hub` 命令：

```python
trainer.push_to_hub(**kwargs)
```

这将把训练日志和模型权重保存在 `"your-username/the-name-you-picked"` 下。有关此示例，请查看位于 `sanchit-gandhi/whisper-small-dv` 的上传。

虽然微调后的模型在 Common Voice 13 Dhivehi 测试数据上取得了令人满意的结果，但它绝非最佳。本指南的目的是演示如何使用🤗 `Trainer` 微调用于多语言语音识别的 ASR 模型。

如果您可以使用自己的 GPU 或订阅了 Google Colab 付费计划，您可以将 `max_steps` 增加到 4000 步，通过更多步骤的训练进一步提高 WER。4000 步训练大约需要 3-5 个小时，这取决于您的 GPU，WER 结果比 500 步训练低大约 3%。如果您决定训练 4000 步，我们还建议您将学习率调度器更改为线性调度器（设置 `lr_scheduler_type="linear"`），因为这将在长时间训练运行中产生额外的性能提升。

通过优化训练超参数，如学习率和dropout率，以及使用更大的预训练检查点（中型或大型），可能会进一步改善结果。我们将此作为一项练习留给读者。

## 分享您的模型

现在，您可以使用 Hub 上的链接与任何人共享此模型。他们可以使用标识符 `"your-username/the-name-you-picked"` 将其直接加载到 `pipeline()` 对象中。例如，加载微调检查点 ["sanchit-gandhi/whisper-small-dv"](https://huggingface.co/sanchit-gandhi/whisper-small-dv)：

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="sanchit-gandhi/whisper-small-dv")
```

## 结论
在本节中，我们逐步介绍了如何微调用于语音识别的 Whisper 模型 🤗 Datasets、Transformer 和 Hugging Face Hub。我们首先加载了 Common Voice 13 数据集的 Dhivehi 子集，并通过计算对数梅尔频谱和tokenize文本对其进行了预处理。然后，我们定义了数据整理器、评估指标和训练参数，然后使用 🤗 `Trainer` 来训练和评估我们的模型。最后，我们将经过微调的模型上传到 Hugging Face Hub，并展示了如何使用 `pipeline()` 类共享和使用该模型。

如果你一直跟进到这里，你现在应该有了一个经过微调的语音识别检查点，干得不错！🥳 更重要的是，您已经掌握了在任何语音识别数据集或领域中微调 Whisper 模型所需的所有工具。还等什么？从 ["选择数据集"](chapter5/choosing_a_dataset.md)部分中选择一个数据集，或者选择一个您自己的数据集，看看您是否能获得最先进的性能！排行榜在等着你...
