# 使用 Gradio 构建 demo
在音频分类的最后一节，我们将制作一个 [Gradio](https://www.gradio.app/) demo，展示我们刚刚在 [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) 数据集上训练的音乐分类模型。首先要做的是使用 `pipeline()` 类加载微调后的检查点--这在[预训练模型](chapter4/pre-trained_models_for_audio_classification.md)部分已经非常熟悉了。您可以将 `model_id` 更改为 Hugging Face Hub 上微调模型的命名空间：

```python
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)
```

其次，我们将定义一个函数，用于获取音频输入的文件路径并将其通过流水线。在这里，流水线会自动加载音频文件，重新采样到正确的采样率，并使用模型运行推理。我们将 `preds` 的模型预测结果格式化为字典对象，并显示在输出上：

```python
def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs
```

最后，我们使用刚才定义的函数启动 Gradio demo：

```python
import gradio as gr

demo = gr.Interface(
    fn=classify_audio, inputs=gr.Audio(type="filepath"), outputs=gr.outputs.Label()
)
demo.launch(debug=True)
```

这将启动一个 Gradio demo，与在 Hugging Face Space 上运行的demo类似。