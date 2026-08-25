# AIC 2026 城市场景多模态目标检测 — H200 训练包

> **给帮忙跑训练的朋友**：感谢帮忙！这个包跑起来只需要 3 步，全程无人值守。
> 跑完后只需执行一条打包命令，把结果文件发回即可。

---

## 0. 这是什么

一个物体检测竞赛（AIC 2026）的训练包。模型输入是 **4 通道图像**（RGB 可见光 3 通道 + 红外灰度 1 通道），12 个类别，评价指标 COCO mAP@50-95。

包内包含两个**顺序执行的实验**（自动跑，无需干预）：

| 实验 | 模型 | 内容 | 预计时长 (H200) |
|------|------|------|----------------|
| A | YOLO26l (26M) | 小图 3× 过采样 + EQLv2 稀有类均衡损失 | 3~6 小时 |
| B | YOLO26x (59M) | 同上（更大容量） | 6~12 小时 |

两实验均为单卡任务，**只需 1 张 H200**（显存占用 < 30GB，141GB 卡绰绰有余）。

## 1. 环境要求

- Linux + NVIDIA H200（Hopper 架构）
- Python 3.9+，`torch >= 2.4`（CUDA 12.4/12.6）
- 磁盘空间：数据集 7.6GB + 过采样数据集 9.1GB + 训练产物 ~5GB ≈ **25GB**

## 2. 三步跑起来

```bash
# 第 1 步：解压上传的包并进入
unzip aic_h200_package.zip    # 或 tar -xzf aic_h200_package.tar.gz
cd aic_h200_package

# 第 2 步：后台启动（推荐，SSH 断开也不影响）
nohup bash run_all.sh > outputs/run_all.log 2>&1 &

# 第 3 步：想起来了就瞄一眼进度（可选）
tail -f outputs/exp_yolo26l_4ch_oversample/progress_log.txt   # 实验A逐轮进度
tail -f outputs/run_all.log                                   # 整体日志
```

**第一次运行** `setup_env.sh` 会自动安装/校验依赖（ultralytics 8.4.116，约 1-2 分钟）。
如果服务器已有 torch 2.4+ 环境，直接复用即可，不会重复安装。

> ⚠️ 注意：`ultralytics` 版本**必须**是 8.4.116（训练脚本依赖其内部接口，其他版本会静默出错）。

## 3. 目录结构

```
aic_h200_package/
├── README.md                # 本文件
├── setup_env.sh             # 环境检查/安装
├── run_all.sh               # 一键全跑（A→B→评估→报告）
├── run_expA.sh / run_expB.sh# 单实验版本
├── pack_results.sh          # ★跑完后执行：打包结果（见第4节）
├── yolo26l.pt / yolo26x.pt  # 预训练权重
├── dataset/                 # 基础数据集（4ch PNG，训练1657+验证343）
├── dataset_oversample/      # 运行时自动生成（小图3x过采样，~2分钟）
├── scripts/                 # 所有训练/监测/评估脚本
├── outputs/                 # 运行时生成：训练产物+日志+评估结果
└── final_report.md          # 跑完后自动生成的汇总报告
```

## 4. ★ 跑完后：打包结果发回（重要）

**只执行这一条命令**：

```bash
bash pack_results.sh
```

它会自动汇总报告并把结果打包成 **`results_back.tar.gz`**（几百 MB ~ 2GB）：

- `final_report.md` — 汇总报告（两个实验的曲线/最优轮次/每类AP/与基准对比）
- 两个实验的 `best.pt`（最终模型）+ `last.pt`（断点）+ `results.csv`（原始曲线）
- `eval_expA.txt` / `eval_expB.txt` — 分组评估（全量/小图/大图）
- `progress_log.txt` — 逐轮监测日志

**把这个 `results_back.tar.gz` 文件发回即可，其余文件（数据集、脚本等）都不需要回传。**

## 5. 运行中发生了什么

1. 环境检查 → 2. 构建过采样数据集（纯文件复制）→ 3. 实验A 训练 100 轮 → 4. 实验A 评估 → 5. 实验B 训练 100 轮 → 6. 实验B 评估 → 7. 生成 final_report.md

每轮训练结束都会往 `outputs/<实验>/progress_log.txt` 追加一行，形如：

```
[08-21 14:33:05] ep  37/100 | mAP50=0.5512 mAP50-95=0.3921 (best ep31=0.3952) | box=1.211 cls=1.543 | lr=0.003250 | elapsed=12h24m | ETA~21h12m
```

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `CUDA 不可用` | 检查驱动 / 是否有 GPU 权限（docker 需要 `--gpus all`） |
| `torch 过旧` | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 显存 OOM（不会发生，141GB 很充裕） | 改小 batch：`python3 scripts/train_yolo26l_oversample.py --batch 4` |
| 实验A中途失败 | **不用担心，会自动继续跑实验B**（run_all.log 记录失败原因，final_report.md 会标明哪个实验未运行） |
| 训练中途机器要重启/被占用 | 没关系：实验用固定 checkpoints，重启后重跑对应脚本即可（已跑完的轮次不会浪费，见下） |
| 想恢复实验A的断点 | `python3 scripts/resume_exp.py --exp expA`（从 last.pt 继续） |
| 只跑实验B | `bash run_expB.sh` |
| SSH 断开导致任务中断 | 务必用 `nohup bash run_all.sh > outputs/run_all.log 2>&1 &` 启动（见第 2 节），前台直接跑会随 SSH 断开被杀 |

## 7. 免责声明

训练数据为竞赛官方提供，仅用于本竞赛。请勿将包内数据用于其他用途。

**再次感谢！** 有任何报错把 `outputs/run_all.log` 尾部截图发来即可。
