# Task7 Pancreas 预处理量化报告

## 结论

MSD Task07 Pancreas 预处理结果已经通过全量量化验收。当前可用于汇报的证据不只包含 overlay 可视化，还包含文件配对、空间几何、spacing 统一、shape/FOV 变化、强度范围、mask 离散性和 mask/image 体素级对应关系。

核心结论：

- 原始训练集 image/label 数量：281 / 281。
- 预处理后 image/label 数量：281 / 281。
- 全量验收结果：281 / 281 通过，0 个失败样本。
- 预处理后 spacing 达到目标 `(0.8027, 0.8027, 2.5)` mm。
- 预处理后 orientation 全部为 `RAS`。
- 预处理后 image 强度范围为 `[0, 1]`。
- 预处理后 label 值严格为 `{0, 1, 2}`，没有小数类别。
- mask 非空样本数：281 / 281。
- mask 与 image 非背景区域重叠比例最小值为 `0.978796`，中位数为 `0.999950`。

## 数据来源

本报告使用当前仓库内已经生成的统计与验收结果：

- 原始数据 header 统计：`data_exam_report/stats_Task07_Pancreas.json`
- 预处理后 header 统计：`data_prep_exam_report/stats_prep_processed_MSD_Task7.json`
- 全量体素级验收：`task7_preprocess_validation/output/summary.json`
- 预处理 pipeline：`data_prep/run_preprocess.py`
- 预处理参数：`data_prep/config_prep.py`

## 预处理流程

当前预处理代码的流程是：

1. `LoadImaged`
   读取 image 和 label 的 NIfTI 文件。
2. `EnsureChannelFirstd`
   确保张量形状变为 channel-first。
3. `Orientationd(axcodes="RAS")`
   统一空间方向到 RAS。
4. `CropForegroundd(source_key="image", select_fn=x > -500, margin=5)`
   基于 CT HU 前景阈值裁掉空气背景，同时对 image 和 label 使用同一个裁剪框。
5. `ScaleIntensityRanged(a_min=-87, a_max=199, b_min=0, b_max=1, clip=True)`
   将 CT 强度窗口归一化到 `[0, 1]`。
6. `Spacingd(pixdim=(0.8027, 0.8027, 2.5), mode=("bilinear", "nearest"))`
   统一 spacing。image 用 bilinear 插值，label/mask 用 nearest 插值。
7. `CastToTyped(keys=["label"], dtype=np.uint8)`
   代码中尝试将 label 转为 `uint8`。
8. `SaveImaged(resample=False)`
   保存预处理后的 image 和 label，保存时不再做第二次重采样。

## 医学图像指标解释

### Shape

`shape = (x, y, z)` 表示体素矩阵尺寸，也就是三维数组每个方向有多少个 voxel。

例如原始数据的中位数 shape 是：

```text
(512, 512, 93)
```

这表示每个样本的中位数体素网格大约有：

```text
512 * 512 * 93 = 24,379,392 voxels
```

预处理后的中位数 shape 是：

```text
(498, 421, 107)
```

中位数体素数量约为：

```text
498 * 421 * 107 = 22,433,406 voxels
```

shape 变化并不等于物理尺寸一定变小或变大，因为还要同时看 spacing。

### Spacing

`spacing = (sx, sy, sz)` 表示每个 voxel 在真实物理空间中的大小，单位是 mm。

原始数据 spacing 范围：

| Axis | Min | Median | Max |
| --- | ---: | ---: | ---: |
| x | 0.605469 | 0.802734 | 0.976562 |
| y | 0.605469 | 0.802734 | 0.976562 |
| z | 0.700012 | 2.500000 | 7.500000 |

预处理后 spacing 范围：

| Axis | Min | Median | Max |
| --- | ---: | ---: | ---: |
| x | 0.802700 | 0.802700 | 0.802734 |
| y | 0.802700 | 0.802700 | 0.802734 |
| z | 2.500000 | 2.500000 | 2.500000 |

这说明原始数据 z 轴 spacing 差异很大，预处理后已经统一到目标 spacing。

### Voxel

voxel 是三维图像里的最小网格单位。一个 voxel 不一定是正方体，它的真实物理大小由 spacing 决定。

例如预处理后的目标 voxel 尺寸约为：

```text
0.8027 mm * 0.8027 mm * 2.5 mm
```

所以 z 方向的一个 voxel 厚度明显大于 x/y 方向。这是 CT 数据常见的各向异性体素。

### FOV

`FOV = shape * spacing`，表示图像覆盖的物理范围。

按中位数估算：

| Stage | Shape median | Spacing median | FOV median approximation |
| --- | --- | --- | --- |
| Raw | `(512, 512, 93)` | `(0.802734, 0.802734, 2.5)` | `(411.0, 411.0, 232.5)` mm |
| Preprocessed | `(498, 421, 107)` | `(0.802700, 0.802700, 2.5)` | `(399.7, 337.9, 267.5)` mm |

预处理后 x/y 方向 FOV 变小，主要来自 `CropForegroundd` 去掉空气背景；z 方向中位数 FOV 变大，是 shape 和 spacing 共同作用的结果，不能只看 shape。

## 原始数据与预处理后对比

| Metric | Raw | Preprocessed | Interpretation |
| --- | ---: | ---: | --- |
| image count | 281 | 281 | 样本未丢失 |
| label count | 281 | 281 | 标签未丢失 |
| geometry mismatch | 0 | 0 | image/label 空间一致 |
| orientation | RAS: 281 | RAS: 281 | 方向统一 |
| shape x median | 512 | 498 | x 方向裁剪后略小 |
| shape y median | 512 | 421 | y 方向空气背景减少明显 |
| shape z median | 93 | 107 | spacing 统一后 z 层数变化 |
| spacing x median | 0.802734 | 0.802700 | 统一到目标 spacing |
| spacing y median | 0.802734 | 0.802700 | 统一到目标 spacing |
| spacing z median | 2.500000 | 2.500000 | 统一到目标 spacing |
| image dtype | N/A | float32: 281 | 训练输入常用浮点 |
| label dtype | N/A | float32: 281 | header dtype 不是 uint8，但 label 值离散正确 |

## 全量验收结果

`task7_preprocess_validation/validate_task7_preprocess.py` 对预处理结果做了全量检查。

验收项目：

- 每个 image 是否存在对应 label。
- image/label 的 shape 是否一致。
- image/label 的 spacing 是否一致。
- image/label 的 affine 是否一致。
- image/label 的 orientation 是否一致。
- spacing 是否达到目标 spacing。
- image 是否有限且落在 `[0, 1]`。
- label 是否只包含整数类别。
- label 是否只包含 `{0, 1, 2}`。
- mask 是否非空。
- mask 是否落在 image 非背景区域内。

验收结果：

| Check | Result |
| --- | --- |
| overall_status | pass |
| sample_count | 281 |
| passed_sample_count | 281 |
| failed_sample_count | 0 |
| pairing_failure_count | 0 |
| target_spacing_reached | true |
| all_orientation_ras | true |
| all_masks_nonempty | true |
| all_images_normalized | true |
| label_values | `[0.0, 1.0, 2.0]` |
| image_min_global / image_max_global | `0.0 / 1.0` |
| mask voxel min / median / max | `13683 / 53595 / 538852` |
| mask-body overlap min / median | `0.978796 / 0.999950` |

## 对老师问题的回答

### “现在是不是主要都是可视化，缺少量化？”

之前 overlay 图确实更偏可视化证据。现在已经补充了量化验收：

- 几何层面：shape、spacing、affine、orientation 全量一致。
- 预处理目标层面：spacing、orientation、image intensity range 全量达标。
- 标签语义层面：label value set 全量为 `{0, 1, 2}`，没有插值产生的小数标签。
- mask/image 对应层面：mask-body overlap 全量统计，最小也达到 `97.8796%`。

### “musk/mask 和 image 能不能对应上？”

可以从两层回答：

1. 空间几何对应：281 对 image/mask 的 shape、spacing、affine、orientation 全量一致。
2. 体素语义对应：mask 体素绝大多数落在 image 非背景区域，mask-body overlap 中位数接近 `99.995%`。

因此没有发现整体错位、翻转、spacing 不一致或 label 插值错误。

## 仍需如实说明的点

预处理代码里有 `CastToTyped(keys=["label"], dtype=np.uint8)`，但当前保存后的 NIfTI label header 统计显示为 `float32: 281`。

这不影响本次“mask 是否对应 image”和“标签语义是否正确”的结论，因为全量体素检查已经确认：

- label 值只有 `{0, 1, 2}`。
- label 值全是整数。
- mask 非空。
- mask 与 image 非背景区域高度重叠。

但如果后续目标是节省存储空间或严格保证 label 文件 header dtype 为 `uint8`，建议单独增加一个 label rewrite 步骤，并再次运行本验收程序。

## 汇报建议

可以这样概括：

> 我把 Task07 预处理从可视化检查扩展成了全量量化验收。现在不仅有 overlay 图，还对 281 个样本逐例检查了 image/label 配对、shape、spacing、affine、orientation、image 强度范围、label 类别集合、空 mask 和 mask 与 image 非背景区域的重叠。结果是 281/281 全部通过，目标 spacing 和 RAS 方向都达成，image 归一化到 `[0,1]`，label 只包含 `{0,1,2}`，mask-body overlap 最小值为 97.88%，中位数为 99.995%。因此当前预处理结果可以作为后续训练输入。
