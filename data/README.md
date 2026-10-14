# PTB-XL数据位置、版本与检查

更新：2026-10-10。数据固定为PTB-XL **v1.0.3**，存放在`D:/datasets/ptb-xl`，不复制到项目或Obsidian库中。本机路径由被Git忽略的`configs/local.yaml`配置；其他机器自行修改。

```text
D:/datasets/ptb-xl/
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/  （元数据引用的 .hea/.dat）
```

本地元数据为21799条记录、18869名患者，100 Hz波形引用43598个`.hea/.dat`文件。历史下载按官方SHA-256核对；后续精确重复审计遍历的是元数据引用的21799个`.dat`，不能将目录里旧版遗留文件作为额外训练样本。

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py --dataset-root D:/datasets/ptb-xl
.venv/Scripts/python.exe scripts/inspect_record.py --ecg-id 1
```

仅在缺少数据时运行`.venv/Scripts/python.exe scripts/download_ptbxl.py --workers 8`；该脚本按配置确定目录并校验已存在文件，可重复运行。100 Hz流程使用`filename_lr`。程序支持500 Hz读取不等于已完成500 Hz模型实验，当前固定输入仍是十二导联100 Hz、10秒。

| 官方集合 | 原始记录 | 当前剔除五类全零后 |
|---|---:|---:|
| folds1–8训练 | 17418 | 17084 |
| fold9验证 | 2183 | 2146 |
| fold10历史测试 | 2198 | 2158 |

先按官方患者划分，再应用标签纳入规则；全零不自动补正常或起搏器。标准化、类别权重和量化校准只使用训练集；阈值和候选选择只使用折9。折10历史反复使用，不作为新的独立盲测。

详细选择理由、分布/共现、质量、标签映射和划分见[完整实验流程](../md/模型训练与优化.md#step-02)。数据不加入Git，下载或复制后核对版本、元数据与波形。官方来源：[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，许可CC BY 4.0，引用与归属见[项目README](../README.md)。
