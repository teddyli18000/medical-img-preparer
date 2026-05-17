# data_exam 数据检查手册

本手册用于说明 `data_exam` 对医学影像数据集到底检查了什么、为什么要检查，以及如何解读结果。当前实现是 **Header-only EDA（仅头信息探索式分析）**，不读取体素数组（voxel array，体素数据）。

## 1. 目标与边界

`data_exam` 的目标是：在训练前快速发现数据配对、文件可读性、空间几何（geometry，几何属性）层面的高风险问题。

已覆盖：

- 文件与标签配对一致性（consistency，一致性）
- NIfTI 头信息可读性（header readability，可读性）
- 体素间距统计（spacing statistics，间距统计）
- 体素维度统计（shape statistics，形状统计）
- 方向编码分布（orientation distribution，方向分布）

未覆盖（不要误解为已检查）：

- 分割标签语义正确性（label semantics，标签语义）
- 标签类别值域（class value range，类别取值范围）
- 前景占比/空标签（foreground ratio / empty mask，前景比例/空掩膜）
- 图像强度分布（intensity distribution，强度分布）

## 2. 执行入口与输出

### 2.1 运行方式

在仓库根目录执行：

```bash
python data_exam\run_exam.py
```

### 2.2 输入来源

由 `data_exam\config_exam.py` 决定：

- `RAW_DATA_DIR`：数据集根目录（应包含 `imagesTr` 与 `labelsTr`）
- `IMAGES_SUBDIR` / `LABELS_SUBDIR`
- `FILE_GLOB`（默认 `*.nii.gz`）

### 2.3 输出文件

输出到 `data_exam_report`：

- `stats_<dataset>.json`：结构化结果，便于程序消费
- `report_<dataset>.md`：人工可读报告

---

## 3. 检查项总览（检查了什么 + 为什么检查）


| 检查项                                                  | 实现位置                                      | 检查内容（What）                    | 检查原因（Why）                                  |
| ------------------------------------------------------- | --------------------------------------------- | ----------------------------------- | ------------------------------------------------ |
| 输入目录存在性（directory existence，目录存在性）       | `analyze_headers.py::_validate_input_dirs`    | 校验`imagesTr`、`labelsTr` 是否存在 | 避免后续统计在错误路径上“成功运行但结果无意义” |
| 文件发现规则（file discovery，文件发现）                | `_list_nii_files`                             | 按`FILE_GLOB` 收集文件，可排序      | 保证样本集合稳定可复现（reproducible，可复现）   |
| 样本 ID 规范化（sample id normalization，样本ID规范化） | `_sample_id_from_path`                        | 去除`.nii.gz/.nii` 后缀用于配对     | 建立 image 与 label 的一一映射基础               |
| 总数一致性（count consistency，数量一致）               | `summary.counts_match`                        | 比较 image 数与 label 数            | 数量不一致通常意味着缺标注或冗余标注             |
| 缺失标签（missing label，缺失标签）                     | `missing_label_sample_ids`                    | 存在 image 但无同名 label           | 训练会出现监督信号缺失，必须修复                 |
| 孤儿标签（orphan label，孤儿标签）                      | `orphan_label_sample_ids`                     | 存在 label 但无同名 image           | 会污染数据管理与统计，可能是命名/拷贝错误        |
| Header 可读取性（header readability，头信息可读取）     | `nib.load` + 异常捕获                         | 对每个 image 读取 NIfTI 头          | 快速识别损坏文件（corrupted file，损坏文件）     |
| Spacing 维度合法性（spacing dimensionality，间距维度）  | `len(zooms) >= 3`                             | 必须至少有 3 个轴向 spacing         | 3D 训练前提；维度不足会导致预处理失败            |
| Shape 维度合法性（shape dimensionality，形状维度）      | `len(shape) >= 3`                             | 必须至少 3 维体素形状               | 防止 2D/异常数据混入 3D 管线                     |
| Spacing 提取（spacing xyz）                             | `header.get_zooms()`                          | 记录 x/y/z 轴体素间距（mm）         | 间距异质性会影响重采样（resampling，重采样）策略 |
| Shape 提取（shape xyz）                                 | `nifti_img.shape`                             | 记录 x/y/z 体素尺寸（voxels）       | 尺寸跨度影响 patch 大小、显存预算、裁剪策略      |
| 方向编码（orientation code，方向编码）                  | `aff2axcodes(affine)`                         | 统计 RAS/LPS 等方向                 | 多方向混杂会造成解剖方向不一致，需标准化         |
| 失败样本明细（failure details，失败明细）               | `failures` + `samples`                        | 记录 sample_id、路径、异常信息      | 支持精确回溯与定点修复                           |
| 统计聚合（aggregate stats，聚合统计）                   | `_compute_spacing_stats/_compute_shape_stats` | 计算 min/max/median                 | 量化数据分布，给预处理参数提供依据               |
| 结论生成（conclusions，结论）                           | `_build_conclusions`                          | 自动生成可执行结论文本              | 让报告从“原始数字”变成“可行动判断”           |

---

## 4. 报告字段解读（如何读结果）

### 4.1 `summary`

- `image_file_count` / `label_file_count`：样本与标注总量
- `counts_match`：总量是否一致
- `missing_label_count` / `orphan_label_count`：配对异常数量
- `header_read_success_count` / `header_read_failed_count`：可读性结果

### 4.2 `stats`

- `spacing_xyz.{x,y,z}.{min,max,median}`：体素间距范围与中位数
- `shape_xyz.{x,y,z}.{min,max,median}`：体素尺寸范围与中位数
- `orientation_distribution`：方向编码分布

### 4.3 `consistency`

- `missing_label_sample_ids`：缺失标签样本清单
- `orphan_label_sample_ids`：孤儿标签样本清单

### 4.4 `samples` 与 `failures`

- `samples`：逐样本完整检查记录
- `failures`：只保留失败项，便于快速排障（troubleshooting，故障排查）

---

## 5. 现有检查结果（仓库内已有报告）

根据 `data_exam_report\report_Task07_Pancreas.md`（生成时间 `2026-04-13T10:39:46`）：

- image 与 label 均为 `281`，`counts_match=true`
- `missing_label_count=0`，`orphan_label_count=0`
- `header_read_failed_count=0`
- `orientation_distribution` 仅 `RAS: 281`
- spacing z 轴范围 `0.700012... ~ 7.5`，shape z 轴范围 `37 ~ 751`

这意味着：配对与可读性正常，但 z 轴间距与层数跨度较大，训练前应认真设计重采样（resampling）与统一体素策略。

## 6. 常见异常与处理建议

- `missing_label_sample_ids` 非空：补齐同名 label，或剔除对应 image。
- `orphan_label_sample_ids` 非空：核对是否命名错误，或删除多余 label。
- `header_read_failed_count > 0`：优先替换损坏文件，避免把失败样本带入训练。
- 方向分布多种并存：在预处理中统一到目标方向（例如 RAS）。
- spacing/shape 范围跨度过大：先定目标 spacing，再做重采样与裁剪参数回推。

## 7. 配置与复用建议

如果你要检查新数据集，最少改动：

1. `RAW_DATA_DIR`
2. 必要时 `FILE_GLOB`（如包含 `.nii`）
3. 可选 `DATASET_NAME`（决定输出文件名）

建议保留：

- `FAIL_ON_MISSING_DIR=True`：防止路径错误被静默忽略
- `SORT_FILES=True`：保证结果稳定
- `SAVE_ABSOLUTE_PATHS=True`：便于跨目录排障
