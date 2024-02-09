# 用 Gradio 制作demo
现在，我们已经为迪维希语语音识别微调了 Whisper 模型，让我们构建一个 Gradio 演示，向社区展示它！

首先要做的是使用 `pipeline()` 类加载微调后的检查点--这在[预训练模型部分](chapter5/pre-trained_models_for_speech_recognition.md)已经非常熟悉了。你可以将 model_id 改为 Hugging Face Hub 上微调模型的命名空间，或者改成预训练的 [Whisper 模型](https://huggingface.co/models?sort=downloads&search=openai%2Fwhisper-)之一，以执行zero-shot语音识别：

```python
from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)
```

其次，我们将定义一个函数，用于获取音频输入的文件路径并将其通过流水线。在这里，流水线会自动加载音频文件，重新采样到正确的采样率，并使用模型运行推理。然后，我们只需将转录文本作为函数的输出返回即可。为了确保我们的模型能够处理任意长度的音频输入，我们将按照[预训练模型部分](chapter5/pre-trained_models_for_speech_recognition.md)]的描述启用*分块*功能：

```python
def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "sinhalese",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]
```

我们将使用 Gradio[块](https://www.gradio.app/docs/interface)功能在演示中启动两个选项卡：一个用于麦克风转录，另一个用于文件上传。

```python
import gradio as gr

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.outputs.Textbox(),
)
```

最后，我们使用刚才定义的两个块启动 Gradio 演示：

```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True)
```

这将启动一个 Gradio demo，与在 Hugging Face Space 上运行的demo类似。

如果您希望在 Hugging Face Hub 上托管您的demo，您可以使用此空间作为您微调模型的模板。

点击链接将模板演示复制到您的账户： https://huggingface.co/spaces/course-demos/whisper-small?duplicate=true

我们建议您给自己的空间取一个与微调模型相似的名字（例如 whisper-small-dv-demo），并将可见性设置为 "公开"。

将空间复制到账户后，点击 "文件和版本"->"app.py"->"编辑"。然后将模型标识符更改为微调后的模型（第 6 行）。滚动到页面底部，点击 "提交更改到主界面"。演示将重新启动，这次使用的是微调后的模型。您可以与亲朋好友分享这个演示，这样他们就可以使用您训练的模型了！

查看我们的视频教程，更好地了解如何复制 Space 👉️ [YouTube 视频](https://www.youtube.com/watch?v=VQYuvl6-9VE)

我们期待在 Hub 上看到您的演示！