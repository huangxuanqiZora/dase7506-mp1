# Kaggle 运行说明（DASE7506 MP1）

需要上传两个文件（都在本文件夹）：

- `mp1_data.zip` —— 数据集（原始 WikiText-2 数据 + 固定评测代码）
- `MP1_Kaggle.ipynb` —— Notebook（包含改进模型、训练脚本和实验套件）

## 第 1 步：把数据上传成 Kaggle Dataset（只需一次）

1. 打开 https://www.kaggle.com/datasets ，点右上角 **New Dataset**。
2. 把 `mp1_data.zip` 拖进去上传（Kaggle 会自动解压）。
3. 标题填 `mp1-wikitext2-data`，其他默认，点 **Create**。
4. 等状态变成 ready。

## 第 2 步：导入 Notebook

1. 打开 https://www.kaggle.com/code ，点 **New Notebook**。
2. 菜单 **File → Import Notebook**，上传 `MP1_Kaggle.ipynb`。
3. 右侧栏 **Add Input** → 搜索并添加第 1 步创建的 `mp1-wikitext2-data`。

## 第 3 步：设置 GPU

1. 右侧栏 **Settings → Accelerator** 选 **GPU T4 x2**（若没有就选 P100）。
2. **Internet** 建议打开（以防需要安装 `tokenizers`）。

## 第 4 步：先跑 smoke

1. 找到写着 `MODE = "smoke"` 的单元格，保持 smoke 不变。
2. 菜单 **Run → Run All**。
3. 等待结束（约几分钟）。把最后打印的 summary 表格发给我。

smoke 会做两件事：复现基线（约 2.10 BPB）并跑一个短的改进模型，用来确认环境正常、并测出 GPU 速度。

## 第 5 步：跑完整实验

确认 smoke 没问题后，把那个单元格改成 `MODE = "full"`，再 **Run All**。
结果会保存在 `/kaggle/working/mp1_results.zip`，可以下载后发我，或直接把 summary 表格发我。

> 提示：实验是**可续跑**的——同一个 Notebook 再运行时会跳过已经完成的实验，不会重复训练。若 Kaggle 会话超时，重新运行即可继续。
