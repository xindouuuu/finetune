# 大模型训练方式速查

## 先分清两个维度

- **训练目标**：模型用什么数据、要学会什么。常见有继续预训练和 SFT。
- **参数更新方式**：训练时改哪些参数。常见有全量微调和 LoRA。

它们不是互斥选项：可以做全量 SFT，也可以做 LoRA-SFT；继续预训练也可以用全量参数或 LoRA。

## 继续预训练

从已经预训练好的模型继续训练，通常使用某个领域的原始文本，例如产品手册、技术文档或行业语料。目标仍是根据前文预测下一个 token。

典型流程：加载预训练模型、清理并分块文本、计算 next-token loss、反向传播、更新参数、保存 checkpoint。若使用全量微调，模型全部可训练参数都会更新；若使用 LoRA，则通常冻结底座，只更新 LoRA 参数。

**适用场景：**让模型更熟悉某领域的术语、表达和文本知识。它不是从零开始训练模型。

## SFT（监督微调）

用有目标答案的示例教模型按指令回应，例如：

```text
用户：手机网络无法连接。
理想回答：可能是网络连接故障，请检查网络配置……
```

典型流程：准备指令与理想回答、应用聊天模板、分词、计算 loss、反向传播并更新参数，再用验证数据检查效果。

SFT 的训练标签会区分各 token 在对话中的位置，例如 system、user、assistant。训练代码还要决定哪些位置计入 loss。常见做法是只对 assistant 回答计算 loss，让问题作为上下文、回答作为学习目标。

**适用场景：**训练指令跟随、对话风格、问答或领域任务回答。

## LoRA 与全量微调

**全量微调：**基础模型的全部可训练参数都参与更新，显存和优化器状态开销较大。

**LoRA：**在选定层增加小型适配参数，通常冻结基础模型，只训练 LoRA 参数；序列分类时也可能训练分类输出头。参数量和保存的适配器通常较小。

LoRA 回答的是“怎样更新参数”，不决定模型要做续写、聊天还是分类。

| 组合 | 学习目标 | 通常更新什么 |
|---|---|---|
| 全量继续预训练 | 领域原文的下一个 token | 基础模型全部可训练参数 |
| LoRA 继续预训练 | 领域原文的下一个 token | LoRA 参数 |
| 全量 SFT | 指令与理想回答 | 基础模型全部可训练参数 |
| LoRA-SFT | 指令与理想回答 | LoRA 参数，某些任务还会更新任务头 |

## 全文 loss 和回答 loss

因果语言模型都可以用“根据前面的 token 预测下一个 token”来训练。这个目标形式常用于预训练，也用于生成式 SFT；**仅凭 next-token loss 不能判断任务是预训练还是 SFT，数据和监督目标也很重要。**

例如对话：

```text
<user> 你叫什么名字？
<assistant> 我叫 Julie。
```

如果代码使用 `labels=input_ids`，模型会对整段序列计算 next-token loss，包括角色标记、用户问题和 assistant 回答。对话格式仍然包含问答结构，但模型也被训练去预测用户问题等 token；这不是只针对回答的 SFT 目标。

常见的 assistant-only SFT 仍把完整对话作为输入上下文，但在 user/system 等位置把 label 设为 `-100`，只对 assistant 回答位置计算 loss。这样模型看得到问题，却不会因没有复述问题而受罚。

- `attention_mask` 控制哪些输入 token 是有效上下文，主要用于处理 padding。
- `labels` 中的 `-100` 控制哪些位置不计入 loss。
- 两者不是一回事。

因此，SFT 会区分提问和回答；是否只训练回答，要看训练数据的标签构造或训练器的 loss mask 设置。

## 仓库脚本对照

| 文件 | 实际用途 | 参数更新与注意事项 |
|---|---|---|
| [pretrain_finetune_gpt2.ipynb](pretrain_finetune_gpt2.ipynb) | GPT-2 在电话说明书纯文本上做领域继续预训练 | 全量参数；有 `zero_grad()`、`backward()`、`optimizer.step()`。传入 `labels=batch_ids` 做整段文本 next-token loss。 |
| [pretrain_finetune_qwen.ipynb](pretrain_finetune_qwen.ipynb) | 加载、推理和探索 Qwen tokenizer/loss | 当前没有实际训练循环。 |
| [SFT_finetune.py](SFT_finetune.py) | GPT-2 用聊天示例训练生成回复 | 全量参数；`train_step` 确实调用优化器更新。使用整段 `labels=X`，没有屏蔽 user/system 部分，因此不是 assistant-only loss。 |
| [lora_finetune_gpt2.py](lora_finetune_gpt2.py) | GPT-2 对话生成 | LoRA；整段对话 next-token loss。 |
| [lora_finetune_phone_gpt2.py](lora_finetune_phone_gpt2.py) | GPT-2 生成手机故障原因和行动计划 | LoRA；整段格式化文本 next-token loss。当前入口连续调用两次 `train()`。 |
| [lora_finetune_emotion_gpt2.py](lora_finetune_emotion_gpt2.py) | GPT-2 六分类情绪识别 | LoRA + 分类头训练；不是生成式聊天 SFT。 |
| [lora_finetune_qwen.py](lora_finetune_qwen.py) | Qwen + TRL `SFTTrainer` + LoRA | 用 Capybara 对话数据做监督微调。 |
| [lora_finetune_qwen.ipynb](lora_finetune_qwen.ipynb) | Qwen 手机日志 SFT | 配置 LoRA、训练集/评估集和按 epoch 评估；自定义 callback 仅打印一个样例及 loss，不等于完整 accuracy/F1 评估。 |
| [SFT_fine_tune_qwen.ipynb](SFT_fine_tune_qwen.ipynb) | 名称表示 Qwen SFT | 手写训练循环把 `optim.zero_grad()` 和 `optim.step()` 注释掉；虽计算梯度，但不会执行参数更新。 |
| [lora_finetune_qwen_phone_01.py](lora_finetune_qwen_phone_01.py) | 测试基础 Qwen 生成 | 只推理，没有训练。 |
| [lora_finetune_qwen_phone_02.py](lora_finetune_qwen_phone_02.py) | 检查手机日志 JSON | 在 `exit()` 处停止，后面的 LoRA Trainer 训练代码不会运行。 |
| [lora_finetune_qwen_phone_03.py](lora_finetune_qwen_phone_03.py) | 加载底座和已训练的 LoRA adapter 后测试生成 | 推理脚本，不做训练。 |

## 训练代码的最小更新流程

```python
optimizer.zero_grad()                 # 清除上一步累计的梯度
outputs = model(input_ids, labels=labels)  # 前向计算和 loss
loss = outputs.loss
loss.backward()                       # 计算梯度
optimizer.step()                      # 真正更新可训练参数
```

`model.train()` 只切换训练模式；它不更新参数。`loss.backward()` 只计算梯度；没有 `optimizer.step()` 就不会用这些梯度更新权重。

## 一句话记忆

- **继续预训练：**给模型读领域原文，继续练习续写。
- **SFT：**给模型看指令和理想回答，教它按要求回应。
- **LoRA：**用较少的新增参数完成微调的一种方式。
- **全量微调：**更新基础模型的全部可训练参数。
- **判断是否真的训练：**检查是否实际执行了 `optimizer.step()`，或使用的 Trainer 是否确实运行了训练。