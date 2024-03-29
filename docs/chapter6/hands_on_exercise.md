# 动手实践

在本单元中，我们探讨了文本到语音的音频任务，讨论了现有数据集、预训练模型以及针对新语言微调 SpeechT5 的细微差别。

正如您所看到的，在资源匮乏的情况下，对文本到语音任务的模型进行微调极具挑战性。同时，评估文本到语音模型也并非易事。

因此，本实践练习的重点是练习技能，而不是达到某个指标值。

这项任务的目标是在您选择的数据集上对 SpeechT5 进行微调。您可以从同一个 `voxpopuli` 数据集中选择另一种语言，也可以选择本单元中列出的任何其他数据集。

请注意训练数据的大小！如果使用 Google Colab 的免费 GPU 进行训练，我们建议将训练数据限制在 10-15 小时左右。

完成微调过程后，请将模型上传到 Hub 共享。确保使用适当的 kwargs 或在 Hub UI 中将模型标记为 `text-to-speech`模型。

请记住，本练习的主要目的是为您提供充分的练习机会，让您完善自己的技能，加深对文本到语音音频任务的理解。