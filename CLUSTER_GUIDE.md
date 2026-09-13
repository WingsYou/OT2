# Six-task study: complete code pipeline, validation and test

这个文件夹已经补齐原始数据预处理、六任务标签、scaler、original12 concept 提取、模型训练和 val/test 四类结果表的代码。默认仍使用已经处理好的数据；只有明确运行 `pipeline.py raw/all` 才重新做原始数据预处理。

完整准备步骤和文件来源见 [PREPARATION.md](PREPARATION.md)。已有结果只需运行 `python -u pipeline.py report`，即可补上 concept 解释与 F1 统计，无需重训。

## 1. 上传和运行

把 `ot2_six_task_study_v1_20260913_complete.tar.gz` 上传到 Cluster 的 `~/`。在**已分配四张 H100 的计算作业**内运行：

```bash
cd ~
tar -xzf ot2_six_task_study_v1_20260913_complete.tar.gz
conda activate ot2
cd ~/ot2_six_task_study_v1_20260913
python -u pipeline.py train --devices 0 1 2 3 \
  --output-root /data/pzhao3/ot2_output/six_task_complete_v1_20260913
```

推荐在你现有的持久会话或调度作业里运行。程序会写日志，不需要额外 `tee`。无需复制到旧 `~/ot2`，无需安装脚本，也无需启动以前的 runner。

如果作业已经设置 `CUDA_VISIBLE_DEVICES`，这里的 `0 1 2 3` 表示该作业可见的四张 GPU。

中断后，在相同目录重新执行相同命令：已完成的模型和缓存直接复用，多模态训练从上一完整 epoch 继续。CBM 模型保存后也直接复用；若中断发生在 CBM 拟合过程中，会重跑该任务的 CBM 拟合，不重训它的多模态模型。

只重新生成汇总表，不训练、不使用 GPU：

```bash
python -u pipeline.py report
# 如汇总上面的新运行：
# python -u pipeline.py report --output-root /data/pzhao3/ot2_output/six_task_complete_v1_20260913
```

## 2. 读取哪些现有文件

默认路径已写在 `study.yaml` 和 `configs/*_full.yaml`。

| Existing source | Default location under `/data/pzhao3/ot2_output/` | Used for |
|---|---|---|
| Original three-task masters | `model_full_assets_v1/inputs/master_<task>.csv` | Existing task labels, official train/val/test splits and artifact paths |
| Full clinical tables and schemas | `clinical/<task>/` and the CSV path recorded in the master | Existing processed clinical features |
| Frozen input scalers | `scalers/train_scalers_<task>.json` | Apply the original training-set statistics |
| Three history-task input configurations | `history_tasks_v1_20260912/inputs/<task>/train_config.yaml` | Existing history-task labels, splits and nonclinical paths |
| Original three-task concept labels | `cbm_strict_v1_20260910/<task>/labels/` | Existing original12 concept labels and definitions, train/val only |
| History-task concept labels | `cbm_history_v1_20260913/strict/<task>/labels/` | Existing original12 concept labels and definitions, train/val only |
| ECG, environment, wearable and retinal artifacts | Paths already recorded in the masters | Read existing tensors/model-view images |

`data_preprocessing_cluster/` 现在已包含完整预处理源码、你提供的 Cluster 配置/scaler，以及 history 和 original12 标签构建代码。`train` 读取上述**已生成的输出**，不需要再次提供原始临床 CSV。重新构建标签或全部数据时，才使用 `pipeline.yaml` 中的原始数据路径。

沿用现有 `ot2` 环境：PyTorch、torchvision、NumPy、pandas、SciPy、scikit-learn、PyYAML。ResNet18 沿用 `IMAGENET1K_V1` 权重，优先使用 PyTorch 已有缓存。如果计算节点没有网络且尚未缓存权重，把现有权重文件路径填入 `study.yaml` 的 `retinal_weights_path`。

## 3. 实际运行内容

任务：MoCA、CVD broad、CVD strict、Osteoporosis history、Kidney-problem history、Lung-problem history。

每个任务重新训练两个独立模型：

- **Full Clinical model**：完整临床视图 + 另外八个模态。
- **Basic Clinical model**：此前约定的 23 列基础临床信息 + 另外八个模态。

`Full model: basic clinical at inference` 是第一个模型的额外推理情形：同时屏蔽非基础 clinical 数值及其观测 mask，其他八个模态保留。它不额外训练一个模型。

Basic23 沿用原范围：年龄相关 3 列、教育 9 列、婚姻状态 5 列、语言背景 6 列。具体字段名单自动保存到各模型的 `input_columns.json`。它不代表仅有年龄和性别，也不代表仅用 clinical 一个模态。

**三个 history 任务的 Full Clinical 定义：**复用已存在的 MoCA full-clinical 表，按当前任务删除它自己的问卷答案和 condition 指示字段。MoCA 视图中已经排除的其他字段不会被重新加入。这是为了在不重做预处理的前提下，提供定义明确的 Full Clinical 对照。

| History task | Own target source removed from the full view | Own condition ID |
|---|---|---:|
| Osteoporosis history | `mhoccur_oa` | 80502 |
| Kidney-problem history | `mhoccur_rnl` | 2005200017 |
| Lung-problem history | `mhoccur_plm` | 4186898 |

多模态训练上限 **30 epochs**，validation AUROC 连续 **7 epochs** 不改善即 early stop；batch size 32，AdamW，学习率 5e-5，seed 42。加载 best validation checkpoint 后评价 val 和 test。不会为了让删除效果更大而选择 checkpoint。

CBM 复用已经核对过的 **original12**：BMI、收缩压、舒张压、HbA1c、HDL、LDL、双眼视力、双眼对比敏感度、自报活动量、睡眠不安。直接读取已有 schema 和实测标签，不新增定义。

```text
Fresh Basic23 + eight-modality model
             ↓ frozen fusion embedding
     predicted clinical concepts
             ↓ 32-unit nonlinear classification head
         downstream prediction
```

CBM 先训练 concept 层，再训练分类头，最后二者联合训练，主分析固定 `lambda=1`。多模态 encoder/fusion 冻结；没有绕过 concepts 的 Basic 或 latent 通路。保留原来的 concept 对齐约束：联合训练只有在各 concept 的 validation loss 不超过 sequential 参考值的 110% 时，才可入选；否则保留 sequential 起点，并记录实际状态。本轮不额外跑 expanded13 和 lambda=10。Sequential warmup 的评价也保存在详情 CSV。

## 4. 四张表

每张表都分别输出 **val / test**，报告 **AUROC、Balanced accuracy、positive-class F1**。所有模型阈值来自各自 validation 的 Youden 规则；删除输入和恢复通道时固定父模型阈值。95% CI 使用 2,000 次按 outcome 分层的配对参与者 bootstrap。

| Table | Contents | Sign / interpretation |
|---|---|---|
| 1 | Full Clinical vs full model with basic-only clinical at inference vs retrained Basic Clinical | Absolute downstream performance |
| 2 | Basic model: remove each of 9 sources, all retinal, all wearable, or all 8 nonclinical sources | Full-input minus removed-input performance; positive means deterioration |
| 3 | Retrained Basic model vs the same model with the primary CBM readout | Absolute downstream performance |
| 4 | Restore the top 3 predicted concept groups after a specified source removal | Restored-channel minus source-removed performance; positive means recovery |

表二使用该任务整个有标签的评价 split。表四使用“被删除模态原本存在”的人群；删除一组模态时，至少其中一个原本存在即可。表四不要求 12 个实测 concepts 全部齐全，因此不能把本轮数值与之前 complete12 子集结果直接当成同一人群比较。

表四先按**新 validation 的 AUROC 恢复量**选 top3，然后在 test 中固定相同 source、concept 组和排序。不会在 test 上重选 top3。七个候选组为 BMI、Blood pressure、HbA1c、Lipids、Visual function、Activity、Restless sleep。

| Task | Source removed for the main top3 table |
|---|---|
| MoCA | All retinal modalities |
| CVD broad | ECG |
| CVD strict | Wearable activity monitoring |
| Osteoporosis history | ECG |
| Kidney-problem history | All wearable modalities |
| Lung-problem history | All retinal modalities |

同时计算全部 12 种 source 删除与 7 组 concepts 的组合，保留在详情 CSV；另外列出恢复六个心血管代谢 concepts 的联合分析，以及此前 validation 提出的 Lung/OCT 联合恢复。

表四恢复的是**同一人的完整输入模型所预测的 concept 通道**，不是把实测值加入模型。该分析定位分类变化经过哪些通道；它本身还不是临床可部署的缺失模态重建算法。汇总也列出恢复前的删除损失，便于判断恢复量的背景。

这些任务的旧 test 结果曾在项目中被查看过，因此这次是重新训练后的 held-out test 评价，不能称为从未接触过的独立确认队列。区间不包括训练 seed 和此前选择任务/假设的额外不确定性。

## 5. 四张 H100 如何分工

1. 四卡分摊参与者，**只计算一次冻结 ResNet18 的眼底 view features**。这一步仍需读取现有图像，但不重做图像预处理。
2. 十二个多模态训练作业排队，最多四个同时运行，每个使用一张卡；空出的 GPU 自动接下一项。每个 Basic 作业完成后接着训练和评价对应 CBM。
3. 四个眼底分支共享缓存，但 view projection、view/asset attention pooling、其他模态 encoder、fusion 和分类头仍分别训练。缓存只保存不再更新的 CNN 输出。
4. LOO 复用新模型的 source tokens，只重算 fusion/head；CBM 复用新 fusion embeddings。Bootstrap 在分配到的 GPU 上批量计算。

`num_workers: 2` 是每个 GPU 进程的数据读取 worker 数，`cpu_threads: 2` 是每个主进程的计算线程数。建议作业同时获配至少 16 个 CPU 核；按实际获配资源在首次启动前调整。四卡的用途是并行独立模型，不是把同一个小模型强行分成四份；缓存 I/O 和小型 CBM 阶段不保证 GPU 利用率始终达到 100%。

代码和实验配置只保存一份运行快照，训练保持每 epoch checkpoint。没有另外的 QC runner、旧 release 校验链或需要你执行的测试套件。已开始的实验如要改变代码或配置，请使用新的 `output_root`，避免不同版本结果混合。

## 6. 结果在哪里

`study.yaml` 保留旧结果路径，便于汇总已完成实验。上面的新训练命令输出到 `six_task_complete_v1_20260913/`；目录结构相同。旧默认输出：

```text
/data/pzhao3/ot2_output/six_task_study_v1_20260913/
├── tables/
│   ├── SUMMARY_val.md
│   ├── SUMMARY_test.md
│   ├── TABLE_1_val.csv ... TABLE_4_val.csv
│   ├── TABLE_1_test.csv ... TABLE_4_test.csv
│   ├── TABLE_4_joint_val.csv / TABLE_4_joint_test.csv
│   └── classification_detail_*.csv / restoration_detail_*.csv
├── logs/                 # Each cache/model job's log
├── cache/                # Shared frozen CNN and existing normalized tensors
├── <task>/
│   ├── full/             # Checkpoints, curves, predictions, input columns
│   ├── basic/            # Checkpoints, LOO predictions and embeddings
│   └── cbm/              # CBM models, concept schema, selection and restoration
├── RUN_CONFIG.json
└── STATUS.json
```

完成后只需把两个 `SUMMARY_*.md` 发回来即可；需要精确数值时带上 `tables/` 整个小目录。参与者级预测留在 Cluster 的各 task 目录，不混进汇总表目录。

本地仅用合成数据检查了处理产物读取、缓存等价性、指标计算和六任务流程；尚未在真实 Cluster 数据或 H100 上执行本轮实验。

## Code map

| File | Responsibility |
|---|---|
| `run.py`, `worker.py` | One-command scheduling across four GPUs and logs |
| `data.py` | Read existing processed artifacts and cache the frozen retinal backbone |
| `train.py` | Fresh full/basic fitting, fixed-threshold val/test evaluation and LOO |
| `concepts.py` | Existing original12 joint CBM and predicted-channel restoration |
| `report.py` | Four English CSV/Markdown tables for each split |
| `common.py`, `study.yaml`, `configs/` | Shared definitions, statistics and input locations |
| `vendor/` | Reused model, dataset reader and CBM libraries |
| `pipeline.py`, `pipeline.yaml` | Explicit preparation / existing-input training / report entry points |
| `prepare_study_inputs.py` | Bind audited history outcomes and train/val original12 labels to processed inputs |
| `data_preprocessing_cluster/` | Raw preprocessing, supplied task config/scaler and audited label/concept builders |
| `report_update/` | Append concept definitions and F1 operating-point details |
