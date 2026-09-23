# 提交清单（DASE7506 MP1）

最终结果：**test BPB = 1.5139**（基线 ≈ 2.10），validation BPB = 1.4968。

需要提交**两个链接**：① 不可变代码仓库；② 对应的 checkpoint 包（无需重训即可评测）。

---

## 第 1 步：把代码放到 GitHub

1. 登录 https://github.com → 右上角 **+ → New repository**
2. 名称随意（如 `dase7506-mp1`），可见性选 **Public**，**不要**勾选初始化 README
3. 在本地解压 `mp1_submission.zip` 到一个文件夹（例如 `dase7506-mp1`）
4. 在该文件夹里依次执行（把 `你的用户名` 换成你的 GitHub 用户名）：

```powershell
git init
git add .
git commit -m "MP1 submission"
git branch -M main
git remote add origin https://github.com/你的用户名/dase7506-mp1.git
git push -u origin main
```

完成后仓库链接形如：`https://github.com/你的用户名/dase7506-mp1`

> checkpoint（`final/checkpoint.pt`，24 MB）已包含在仓库里，小于 GitHub 单文件 100 MB 限制，可以直接 push。若嫌大，也可以把它单独传到 Kaggle Dataset / 网盘，再把链接作为「checkpoint 包链接」提交。

## 第 2 步：在课程网站提交

打开 https://xudongwu-0.github.io/courses/dase7506/#submit ，填写：

- **Student ID**：<请填写你的学号>
- **Full-test BPB**：`1.5139`
- **Code link**：你的 GitHub 仓库链接
- **Checkpoint link**：仓库里的 `final/checkpoint.pt`（或单独托管的链接）
- 按页面提示完成生成的 GitHub Issue

## 第 3 步：确认

- 截止时间：**2026 年 9 月 30 日（UTC+8）**
- 代码仓库里已包含：`REPORT.md`（≤10 页报告）、`README.md`（含 AI 使用披露）、完整复现说明
- 评测命令（评审者用）：
  ```powershell
  cd code
  python evaluate.py --checkpoint ../final/checkpoint.pt --device cpu --precision fp32 --split test
  ```

---

## 文件说明

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 报告（方法、对比、消融、分析） |
| `README.md` | 安装 / 训练 / 评测说明 + AI 披露 |
| `code/student.py` | 最终模型 |
| `code/train_student.py` | 训练脚本 |
| `code/configs/final.json` | 最终配置 |
| `final/checkpoint.pt` | 冻结的最终模型（提交用） |
| `final/test_cpu_fp32.json` | 测试集结果 |
| `results/` | 各实验汇总表 |
| `kaggle/` | Kaggle Notebook 与运行说明（实验证据） |
