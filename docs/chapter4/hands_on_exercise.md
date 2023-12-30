# 动手实践
现在是时候动手操作一些音频模型，并应用迄今为止所学到的知识了。

以下是说明。在本单元中，我们演示了如何在 `marsyas/gtzan` 数据集上微调用于音乐分类的 Hubert 模型。我们的示例达到了 83% 的准确率。你们的任务是提高这一准确率指标。

请在 [Hugging Face Hub](https://huggingface.co/models) 上选择任何你认为适合音频分类的模型，并使用完全相同的数据集 [`marsyas/gtzan`](https://huggingface.co/datasets/marsyas/gtzan) 来构建你自己的分类器。

您的目标是使用您的分类器在该数据集上达到 87% 的准确率。您可以选择完全相同的模型，并调整训练超参数，也可以选择完全不同的模型--这取决于您！

别忘了在训练结束时将你的模型推送到 Hub，就像本单元中使用以下 `**kwargs` 所显示的那样：

```python
kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

trainer.push_to_hub(**kwargs)
```

以下是一些额外的资源，您可能会发现它们对完成本练习很有帮助：

+ [Transformer文档中的音频分类任务指南](https://huggingface.co/docs/transformers/tasks/audio_classification)
+ [Hubert模型文档](https://huggingface.co/docs/transformers/model_doc/hubert)
+ [M-CTC-T 模型文档](https://huggingface.co/docs/transformers/model_doc/mctct)
+ [音频频谱图Transformer文档](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
+ [Wav2Vec2 文档](https://huggingface.co/docs/transformers/model_doc/wav2vec2)

欢迎制作模型演示，并在 Discord 上分享！如果您有问题，请在 #audio-study-group 频道中提出。