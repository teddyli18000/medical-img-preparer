# data_prep 离线预处理手册（Offline Preprocessing Manual，离线预处理手册）

本手册仅覆盖 `data_prep` 目录内的内容（`config_prep.py` 与 `run_preprocess.py`），以代码**实际行为**为准说明：

- 做了哪些处理（Transform，变换）
- **检查了什么**（Validation/Assumption，校验/假设）
- **为什么要检查**（Rationale，原因）
- 出错时如何定位（Troubleshooting，排障）

> 适用数据：以 CT（Computed Tomography，计算机断层扫描）+ NIfTI（Neuroimaging Informatics Technology Initiative，医学影像文件格式，常见扩展名 `.nii.gz`）为主，且图像强度单位为 HU（Hounsfield Unit，亨氏单位）。

---

## 1. 术语速查（Glossary，术语表）

- Offline preprocessing（离线预处理）：在训练/推理前，把原始数据统一做一次标准化处理并落盘保存。
- Transform（变换）：对数据做的某一步处理，例如重采样、裁剪、归一化。
- Pipeline / Compose（流水线/组合）：把多个 Transform 按顺序串起来执行。
- NIfTI（医学影像格式）：常见 `.nii` / `.nii.gz`。
- Voxel（体素）：三维图像的“像素”。
- Spacing / pixdim（体素间距/物理间距）：每个体素在物理空间的尺寸（mm）。
- Resampling（重采样）：把数据插值到指定 spacing。
- Interpolation（插值）：连续化重采样方法；图像常用 bilinear（双线性），标签常用 nearest（最近邻）。
- Orientation（方向）：医学影像坐标轴方向；本模块统一到 RAS（Right-Anterior-Superior，右-前-上）。
- Foreground（前景）：非空气/非背景区域；此处用阈值（threshold，阈值）来判定。
- Normalization / Scaling（归一化/缩放）：把强度映射到固定范围（本模块映射到 [0, 1]）。

---

## 2. 模块目标与边界（Scope，范围）

### 2.1 目标（Goal，目标）

`data_prep` 负责将原始数据做一次 **确定性（deterministic，可复现）**的离线预处理并保存到硬盘：

1. 统一加载格式（Load）
2. 统一通道维（Channel）
3. 统一方向（Orientation = RAS）
4. 裁剪前景（CropForeground，去空气）
5. 强度归一化（ScaleIntensityRanged，把 HU 映射到 [0,1]）
6. 统一 spacing（Spacingd，三维重采样）
7. 压缩标签 dtype（Cast label to uint8）
8. 输出到 `ConfigPrep.OUTPUT_DIR`

### 2.2 不做什么（Non-goals，不包含）

- 不做训练相关的数据增强（Augmentation，数据增强）
- 不做随机裁剪（Random crop，随机裁剪）
- 不生成数据划分（Split，训练/验证划分）
- 不处理除 `imagesTr`/`labelsTr` 之外的目录结构（例如 `imagesTs`）

---

## 3. 输入数据要求（Input Contract，输入约定）

### 3.1 目录结构检查（Manual check，需人工检查）

脚本在 `ConfigPrep.RAW_DATA_DIR` 下使用 glob（通配符匹配）查找：

- `imagesTr\*.nii.gz`
- `labelsTr\*.nii.gz`

你需要确认原始数据满足：

```
RAW_DATA_DIR\
  imagesTr\  (NIfTI, .nii.gz)
  labelsTr\  (NIfTI, .nii.gz)
```

**为什么要检查：**

- 脚本只会在这两个固定子目录找文件；路径或文件后缀不符合会导致“找不到文件”（空列表）或在加载时报错。

### 3.2 图像-标签配对检查（Pairing check，需人工检查）

脚本配对方式：

```python
img_files = sorted(glob(...imagesTr/*.nii.gz))
lbl_files = sorted(glob(...labelsTr/*.nii.gz))
data_dicts = [{"image": i, "label": l} for i, l in zip(img_files, lbl_files)]
```

你必须人工确认：

1. `imagesTr` 与 `labelsTr` **文件数量一致**
2. 排序后两边的**文件名一一对应**（例如 `pancreas_001.nii.gz` 对 `pancreas_001.nii.gz`）

**为什么要检查：**

- `zip(img_files, lbl_files)` 会以**较短列表为准**截断；若数量不一致，多出来的文件会被**静默忽略**，不会自动报错。
- 若文件名排序无法对齐（例如一边有前缀差异），会把错误的 image 与 label 配到一起，后续裁剪/重采样将产生错误标签对齐。

### 3.3 物理一致性检查（Geometry check，强烈建议人工检查）

建议你额外确认：

- image 与 label 的空间几何一致（Affine/Spacing/Origin/Direction 一致或可正确映射）

**为什么要检查：**

- 预处理会对 image 和 label 执行相同的 Orientation/Spacing/Crop。若原始数据本身未对齐，会把错误对齐“固定下来”，后续再难排查。

### 3.4 强度单位检查（HU check，需人工检查）

本模块默认把 image 当作 CT-HU：

- 前景阈值：`x > -500`（用于裁剪）
- 强度映射窗口：`HU_MIN=-87, HU_MAX=199`（用于归一化）

**为什么要检查：**

- 若数据不是 HU（例如 MRI、或 CT 但已做过不同缩放），`-500` 阈值与 `[HU_MIN, HU_MAX]` 会失效：可能裁剪不到人体或归一化范围不合理。

---

## 4. 配置文件说明（config_prep.py）

位置：`data_prep\config_prep.py`

### 4.1 必填项（Required，必须配置）

- `RAW_DATA_DIR`：原始数据根目录（见第 3 节结构）。

**检查点：**路径是否存在（Path existence，路径存在性）。

- 脚本不会在启动时显式校验该路径；但后续 glob 结果为空时将导致“看似运行但没有产出”（因为 `data_dicts` 为空）。

### 4.2 输出目录（OUTPUT_DIR）

- 默认输出到：`<项目根目录>\data\preprocessed\processed_MSD_Task7`

**检查点：**输出目录可写（Write permission，写权限）与磁盘空间（Disk space，磁盘空间）。

- 脚本会 `os.makedirs(OUTPUT_DIR, exist_ok=True)` 自动创建目录。
- 保存阶段若无权限或空间不足，会在写文件时报错。

### 4.3 空间参数（SPACING）

- `SPACING = (0.8027, 0.8027, 2.5000)`

含义：目标 spacing（target voxel spacing，目标体素间距），单位通常是 mm。

**为什么要检查：**

- spacing 统一能减少不同扫描协议导致的尺度差异，使模型特征更稳定。
- 这是三维重采样最耗时步骤，错误的 spacing 会造成几何失真或不必要的计算量。

### 4.4 强度窗口（HU_MIN/HU_MAX）

- `HU_MIN=-87, HU_MAX=199`

**为什么要检查：**

- 该范围决定了归一化后的对比度；超出范围将 clip（裁切）到 0 或 1。
- 若任务或器官不同，可能需要调整窗口。

### 4.5 并行加载（PREPROCESS_NUM_WORKERS）

- Windows 默认 `0`（单进程），用于规避多进程 pickling（序列化）相关问题。

**为什么要检查：**

- 多进程可以加速 I/O 与预处理，但在 Windows 上更容易出现不可序列化对象导致的报错。

---

## 5. 运行方式（run_preprocess.py）

入口脚本：`data_prep\run_preprocess.py`

### 5.1 推荐命令（Recommended command，推荐命令）

在项目根目录执行：

```powershell
python data_prep\run_preprocess.py
```

### 5.2 运行时会发生什么（Runtime behavior，运行时行为）

- 自动创建输出目录（若不存在）
- 扫描 `imagesTr` 与 `labelsTr` 下的 `.nii.gz`
- 逐对执行预处理流水线并保存
- 控制台输出进度：`已完成: i/N`
- 额外行为：脚本会将 print 输出镜像到项目根目录 `logs` 目录下的文本日志（log file，日志文件）

---

## 6. 预处理流水线逐步说明（Step-by-step Pipeline）

以下步骤顺序与 `run_preprocess.py` 中 `Compose([...])` 完全一致。

### Step 0：创建输出目录（Create output dir，创建输出目录）

- 行为：`os.makedirs(ConfigPrep.OUTPUT_DIR, exist_ok=True)`
- 检查点：目录是否可创建/可写
- 为什么：避免因为路径不存在导致保存阶段直接失败

### Step 1：LoadImaged（Load image/label，加载影像/标签）

- keys：`image`, `label`
- 检查点（运行时）：
  - 文件存在性与可读性（file exists/readable）
  - NIfTI 头信息可解析（header parse）
- 为什么：后续所有空间/强度操作都依赖正确读取的数组与元数据

### Step 2：EnsureChannelFirstd（Ensure channel first，确保通道在最前）

- keys：`image`, `label`
- 检查点（运行时）：输入维度是否可被解释为“带/不带通道”的医学影像
- 为什么：统一张量形状，使后续 Transform 行为一致

### Step 3：Orientationd（Orientation = RAS，统一方向到 RAS）

- keys：`image`, `label`
- 参数：`axcodes="RAS"`
- 检查点（运行时）：是否存在足够的方向信息用于重排轴向
- 为什么：不同数据集可能用不同轴向约定；统一到 RAS 可避免左右/前后/上下颠倒造成的训练混乱

### Step 4：CropForegroundd（Crop foreground，裁剪前景/去空气）

- keys：`image`, `label`（用 image 计算前景框，同时裁剪 label）
- 参数：
  - `source_key="image"`
  - `select_fn = (x > -500)`（阈值）
  - `margin = 5`（保留边缘余量）
- 检查点（核心假设）：image 强度为 HU 且人体组织通常 > -500 HU
- 为什么：
  - 大幅减少空气背景体素，降低后续三维重采样的 CPU/内存成本
  - `margin` 防止裁剪过紧导致边界信息丢失

### Step 5：ScaleIntensityRanged（Intensity scaling，强度缩放到 [0,1]）

- keys：`image`
- 参数：
  - `a_min=HU_MIN, a_max=HU_MAX`
  - `b_min=0.0, b_max=1.0`
  - `clip=True`
- 检查点（核心假设）：HU 范围合理且覆盖主要组织对比
- 为什么：
  - 统一强度分布，提升模型训练稳定性
  - `clip=True` 避免异常高/低值对范围造成影响

### Step 6：Spacingd（Resampling to target spacing，重采样到目标 spacing）

- keys：`image`, `label`
- 参数：
  - `pixdim=SPACING`
  - `mode=("bilinear", "nearest")`
- 检查点（运行时/假设）：
  - 元数据中存在当前 spacing
  - label 使用 nearest 插值（避免类别值被插值成非整数）
- 为什么：
  - 统一物理尺度，减少不同扫描间距带来的分辨率差异
  - 图像用 bilinear 保持平滑；标签用 nearest 保持离散类别

### Step 7：CastToTyped（Cast label dtype，压缩标签类型）

- keys：`label`
- 参数：`dtype=np.uint8`
- 检查点（需人工评估）：标签最大类别值是否 <= 255
- 为什么：
  - 显著减少标签文件体积，提高 I/O 吞吐
  - 若类别数很多（>255），会发生溢出（overflow，溢出），需改 dtype

### Step 8：SaveImaged（Save to disk，保存到硬盘）

- 对 image 与 label 分别保存（两次 SaveImaged）
- 参数要点：
  - `output_dir=OUTPUT_DIR`
  - `output_postfix="image_prep" / "label_prep"`
  - `output_ext=".nii.gz"`
  - `resample=False`（保存时不再二次重采样）
  - `separate_folder=True`（按 key 分目录保存）
- 检查点（运行时）：
  - 输出路径可写
  - 文件名可生成且不冲突
- 为什么：
  - 将离线预处理结果落盘供后续流程使用

---

## 7. 输出结构（Output Layout，输出结构）

输出根目录：`ConfigPrep.OUTPUT_DIR`

文件命名规则（基于 MONAI SaveImaged 的 postfix 机制）：

- 原文件名基础上追加 `_{postfix}`（例如 `_image_prep`）并保持 `.nii.gz` 扩展名。

目录结构（`separate_folder=True` 的行为）：

- SaveImaged 会按 key 创建子目录分别保存（常见为 `image\` 与 `label\`）；实际子目录名以 MONAI 实现为准。

你应检查输出是否满足：

- 每个输入 pair 都产出一对文件（image_prep 与 label_prep）
- 输出的 image/label 在空间上仍对齐（可视化检查或用工具对比元数据）

---

## 8. 运行前检查清单（Pre-flight Checklist，起跑线检查）

> 下面每项都写明“检查什么 / 为什么检查 / 不通过会怎样”。其中**标注为【脚本不会自动检查】**的项，需要你在运行前人工确认。

1. RAW_DATA_DIR 路径存在（Path exists，路径存在）【脚本不会自动检查】

   - 为什么：glob 为空会导致无数据可处理，表现为“运行很快结束且无输出”。
2. `imagesTr`/`labelsTr` 目录存在且含 `.nii.gz` 文件【脚本不会自动检查】

   - 为什么：脚本只匹配该后缀；不匹配将导致数据列表为空。
3. image 与 label 数量一致【脚本不会自动检查】

   - 为什么：`zip` 截断会静默漏处理。
4. image 与 label 文件名可正确一一配对【脚本不会自动检查】

   - 为什么：配错对会把错误标签固化到输出。
5. image 强度单位为 HU（HU scale，HU 标尺）【脚本不会自动检查】

   - 为什么：前景阈值与窗口归一化都基于 HU 假设。
6. 标签类别范围 <= 255【脚本不会自动检查】

   - 为什么：标签会被 cast 成 uint8，超范围会溢出。
7. 输出目录可写且磁盘空间充足（Writable & enough space，可写且空间足）

   - 为什么：SaveImaged 写入失败会中断。
8. 运行环境依赖已安装（Dependencies installed，依赖已安装）

   - 关键依赖：`monai`, `numpy` 以及读取 NIfTI 所需的后端库（通常由 MONAI 依赖引入）。
   - 为什么：缺依赖会在 import 或 LoadImaged 时直接报错。

---

## 9. 常见问题与定位（Troubleshooting，排障）

### 9.1 “运行结束但没有任何输出文件”

优先检查：

- `RAW_DATA_DIR` 是否正确
- `imagesTr`/`labelsTr` 下是否真的有 `*.nii.gz`

原因：`data_dicts` 为空时，DataLoader 不会迭代任何样本。

### 9.2 image/label 数量不一致导致漏处理

现象：只处理了 `min(len(imagesTr), len(labelsTr))` 对。

处理：运行前先人工核对数量与文件名；必要时把配对策略改为“按 basename 映射”（但这属于代码改动，不在本手册范围）。

### 9.3 裁剪结果异常（过度裁剪/几乎不裁剪）

现象：输出尺寸非常小或几乎与原图相同。

原因：

- `x > -500` 阈值不适用于你的数据强度分布。

处理：确认是否为 CT-HU；若不是，需要调整阈值与窗口参数。

### 9.4 标签出现非整数或边界毛刺

原因：标签重采样必须使用 nearest；本脚本已固定为 nearest。

若仍异常：优先怀疑原始 image/label 几何未对齐或配对错误。

---

## 10. 复现性与性能说明（Reproducibility & Performance）

- 本模块无随机 Transform，因此相同输入与参数应输出相同结果（deterministic）。
- 性能关键点：
  - 先 CropForeground 再 Spacing（先瘦身再重采样）能显著降低重采样耗时与内存。

---

# pre
每个样本的体素个数（Shape）不一样，
Same：
物理间距(Spacing) (中位数)
空间方向(Orientation) (RAS)
像素值域(Intensity Range) [0,1]
数据格式(Data Format & Types)   image(C,X,Y,Z), C=1     label uint8