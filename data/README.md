# PTB-XL 数据

PTB-XL 1.0.3 数据集存放在 `D:\datasets\ptb-xl`，位于 Obsidian 仓库之外，避免 Obsidian 扫描数万条波形文件。项目通过被 Git 忽略的 `configs/local.yaml` 指向本机数据位置。100 Hz 流程至少需要：

```text
D:\datasets\ptb-xl\
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/
```

从项目根目录执行 `python scripts/download_ptbxl.py --workers 64`，可从官方公开存储下载 100 Hz 子集，并用官方 `SHA256SUMS.txt` 逐文件校验；默认位置取自 `configs/local.yaml` 或 `configs/default.yaml`。下载脚本可重复运行，已验证的文件会跳过。随后执行 `python scripts/prepare_data.py` 生成官方 fold 清单，并用 `python scripts/inspect_record.py --ecg-id 1` 查看一条记录的波形和信号分析图。

当前工作区已下载 PTB-XL 1.0.3 的完整 `records100`：21,799 条记录、43,598 个波形文件，逐文件 SHA-256 校验通过。数据现位于 `D:\datasets\ptb-xl`。原始数据、SHA 清单和生成的图不会加入 Git；在其他机器克隆项目后，需要重新下载或复制数据，并按本机位置设置 `configs/local.yaml`。

程序也会尝试识别 `data/ptbxl/` 以及数据目录下的一层标准解压子目录。默认使用 100 Hz；如需 500 Hz，需同时提供 `records500/` 并用 `--sampling-rate 500`。官方数据页：[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)。数据文件采用 CC BY 4.0；使用或再分发时请遵循 PhysioNet 页面列出的许可与引用要求。
