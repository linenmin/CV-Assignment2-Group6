# Kaggle 提交一步一步指南

## 提交件清单

最终要上传的有 **两份**：

1. **Dataset (heavy artifacts)**
   - 本地: `kaggle_submission/outputs/kul-cv-ga2-group6.zip` (1421 MB, 43 files)
   - 上传后 Kaggle slug: `kul-cv-ga2-group6`
   - SHA-256: `b5c5366e79a78b9ed0f636cd02b9db749f5ad39a5599923095b862337d845323`

2. **Notebook (report)**
   - 本地: `kaggle_submission/notebook/group6_final.ipynb` (82 cells)

注意：notebook cell ch1-setup 里 `SUBMISSION_DATASET_SLUG = "kul-cv-ga2-group6"` 写死了这个 slug。**上传 dataset 时一定用这个 slug**，否则 notebook 在 Kaggle 上跑会找不到文件。

---

## 一、上传 Dataset 到 Kaggle

### 方法 A：Kaggle CLI（推荐，1.4 GB 用 Web UI 容易超时）

#### 1. 安装 Kaggle CLI

```bash
pip install kaggle
```

#### 2. 获取 API token

1. 登录 https://www.kaggle.com/settings
2. 找到 **API** 部分
3. 点 **Create New Token** → 下载 `kaggle.json`
4. 把 `kaggle.json` 放到 `C:\Users\<你的用户名>\.kaggle\kaggle.json`（Windows）或 `~/.kaggle/kaggle.json`（macOS / Linux）
5. Windows PowerShell 设权限（macOS / Linux 跳过）：
   ```powershell
   icacls "$env:USERPROFILE\.kaggle\kaggle.json" /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```

#### 3. 准备 dataset metadata

```bash
cd kaggle_submission/dataset
kaggle datasets init -p .
```

这会生成 `dataset-metadata.json`。打开它，改成下面的内容（把 `<your-kaggle-username>` 改成你的 Kaggle 用户名）：

```json
{
  "title": "KUL CV GA2 Group 6",
  "id": "<your-kaggle-username>/kul-cv-ga2-group6",
  "licenses": [{"name": "CC0-1.0"}]
}
```

⚠️ `id` 后半部分**必须是** `kul-cv-ga2-group6`，slug 名一字不差，否则 notebook 在 Kaggle 上跑会找不到文件。

#### 4. 创建 dataset

```bash
kaggle datasets create -p . --dir-mode zip
```

`--dir-mode zip` 让 CLI 把整个 `dataset/` 打成 zip 上传，比一个个传文件快得多。1.4 GB 上传时间取决于网速：100 Mbps 上行约 2 分钟。

上传成功后控制台会显示 dataset URL 类似：
```
https://www.kaggle.com/datasets/<your-username>/kul-cv-ga2-group6
```

### 方法 B：Web UI

1. 打开 https://www.kaggle.com/datasets
2. 点右上 **+ New Dataset**
3. 拖拽 `kaggle_submission/outputs/kul-cv-ga2-group6.zip` 进上传区
4. **Title**: `KUL CV GA2 Group 6`
5. **Slug**: 必须改成 `kul-cv-ga2-group6`（默认会基于 title 自动生成，要手动改）
6. **License**: `CC0: Public Domain` 即可
7. 点 **Create**

⚠️ 1.4 GB 经常 Web UI 上传中途断。Wi-Fi 不稳定推荐用 CLI。

### 上传后验证

打开 dataset 页面，确认能看到这些文件夹：
- `code/` (7 个 .py)
- `figures/` (3 个 .svg + 3 个 .png)
- `predictions/`（空文件夹）
- `submissions/` (2 个 .csv)
- `tables/` (17 个 .csv)
- `weights/convnext_small_320/` (1 个 .pth)
- `weights/eomt_dinov3_v17_merge_val_lr1e5/best/` (1 个 .safetensors + 3 个 .json)

---

## 二、上传 Notebook 到 Kaggle

### 1. 进入 Kaggle 比赛页面

打开 KUL CV GA2 比赛的 Kaggle 链接（Toledo 上提供的，类似 `https://www.kaggle.com/competitions/kul-computer-vision-ga-2-2026`）。

### 2. 创建 Notebook

点 **Code → + New Notebook**。

### 3. 在新 notebook 里 import 我们的两个 datasets

右侧面板有 **Input**：

a) 点 **+ Add Input → Competitions**，搜 `kul-computer-vision-ga-2-2026` 加进来（比赛数据）。

b) 点 **+ Add Input → Datasets**，搜 `kul-cv-ga2-group6` 加进来（我们刚上传的 dataset）。

加完后右侧 Input 区应该显示两个挂载点：
- `/kaggle/input/kul-computer-vision-ga-2-2026/`
- `/kaggle/input/kul-cv-ga2-group6/`

### 4. 替换 notebook 内容

a) 在 Kaggle notebook 里点 **File → Import Notebook**，选本地的 `notebook/group6_final.ipynb`。

b) 或者直接复制粘贴 cell。

### 5. 检查 Settings

Kaggle notebook 右侧 **Settings**：
- **Accelerator**: `GPU T4 x2`（或任意 GPU）— 实际上不需要 GPU（我们 cell 都不跑训练），但加速安装包
- **Internet**: 关闭（Kaggle 提交规则）
- **Persistence**: 保留默认

### 6. Run All

点顶部 **Run All**。Kaggle 会重新执行所有 code cell。预计 1-2 分钟（因为 notebook 只是加载 CSV / 画图 / 写 submission.csv，没有训练或推理）。

执行完后底部应该有：
- ch1-setup 输出：`Running on : Kaggle`
- 所有图表、表格都重新渲染
- ch6-write 输出：`Output CSV : /kaggle/working/submission.csv`，`Total rows : 1500`
- ch6-sanity-code 输出：`All five sanity-check assertions passed.`

### 7. Save Version

点右上 **Save Version → Save & Run All (Commit)**。Kaggle 会再跑一遍 notebook 并保存这次结果。

---

## 三、提交到比赛 leaderboard

### 1. 等 Save Version 完成

通常 5-10 分钟。完成后可以看到 notebook 的 "Output" 标签页里有 `submission.csv`。

### 2. 提交

进入比赛页面 **Submit Predictions**：
- 选 **Submit from a Notebook Output**
- 选刚才那个 notebook 的 latest version
- 选 `submission.csv` 文件
- 点 **Submit**

或者直接：
- 在 notebook 输出标签页找到 `submission.csv`
- 点旁边的 **Submit to Competition**

### 3. 验证 leaderboard 分数

Kaggle 会自动评分。我们的目标分数：**0.89139** (joint Dice, 1500-row CSV)。

---

## 四、Share notebook 给 TA

Toledo 上要求把 notebook 分享给 TA（Kaggle Didactic Team 的用户名）：

1. 在 notebook 右上点 **Share**
2. 进入 **Collaborators**
3. 加入 TA 的 Kaggle 用户名（Toledo 上有列表，类似 `cv_ta_2026` 之类）
4. 权限设为 **Can View**
5. 点 **Save**

⚠️ 注意：TA 看 notebook 必须等到 **比赛截止后**。提交前给 TA View 权限是 OK 的。

---

## 故障排查

### 在 Kaggle 上跑 notebook 报 `ModuleNotFoundError: No module named 'data_overview'`

`ch1-setup` 没把 dataset 的 `code/` 加进 `sys.path`。检查：
- Dataset slug 是 `kul-cv-ga2-group6` 吗？
- `Settings → Input` 里 dataset 是不是真的 attach 了？

### `FileNotFoundError: ... segmentation_v17_final_merged.csv`

检查：
- Dataset 上传完整吗？打开 Kaggle dataset 页面，确认 `submissions/submission_v17_final_merged.csv` 存在（1500 行 + 1 header = 1501 行）

### `submission.csv` 行数不对

正常应该是 **1500 行 + 1 header = 1501 lines**。如果 Kaggle 评分提示行数不对，重新检查 `ch6-sanity-code` 的输出，所有五条 assertion 应该全 pass。

### Submitted 后分数和预期 0.89139 不一致

`submission_v17_final_merged.csv` 在 dataset 里就是 0.89139 那一份。如果分数变了，可能是 Kaggle 重跑 notebook 时 ch6-write cell 写出了不一样的 CSV。检查 ch6-write 的输出，行数应该是 1500。
