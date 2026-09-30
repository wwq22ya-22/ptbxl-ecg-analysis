# GitHub 发布与小组查看说明

日期：2026-09-30

## 仓库

- 仓库地址：[wwq22ya-22/ptbxl-ecg-analysis](https://github.com/wwq22ya-22/ptbxl-ecg-analysis)
- 当前可见性：私有。
- 默认分支：`main`。
- 本机项目：`D:\MyProject\college\ecg1`。
- 初始发布提交：`45f65a4c21fd7e4d6f779042de091444b0f591db`；已经核对该提交在 GitHub main 分支存在。

## 让组员查看

仓库所有者登录 GitHub 后，在仓库的 **Settings → Collaborators → Add people** 中填写组员的 GitHub 用户名并发送邀请。组员接受邀请后，即可查看仓库和克隆代码。个人账号私有仓库的协作者可以参与提交，因此添加前应确认对应账号确属小组成员。尚未获得访问权限的用户打开私有仓库链接时可能看到 404。

如果需要所有人直接通过链接查看，可由所有者在仓库设置中明确更改为公开。

## 克隆与运行

```powershell
git clone https://github.com/wwq22ya-22/ptbxl-ecg-analysis.git
cd ptbxl-ecg-analysis
```

后续环境安装、数据下载和启动步骤见仓库根目录的 [README](../README.md)。代码、两种正式模型、ONNX 导出、正式 PTB-XL 指标与训练图表、LUDB 指标和验证报告已经包含在仓库中。原始 PTB-XL / LUDB ECG 数据由每位成员按官方来源另行获取。

旧报告中的 `D:\...` 是开发机器位置。成员应以自己的克隆目录、`configs/default.yaml` 和不提交的 `configs/local.yaml` 为准。

## 本机更新上传

在本机项目根目录修改文件后：

```powershell
git status
git add .
git commit -m "说明本次修改"
git push
```

小组成员开发时建议各自使用分支，通过 Pull Request 合入 main，便于查看修改记录。

## 文件与清理位置

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 本机 Git 仓库 | `D:\MyProject\college\ecg1\.git` | 保留，用于记录历史和同步 GitHub |
| 项目首页 | `D:\MyProject\college\ecg1\README.md` | GitHub 打开仓库时显示 |
| 发布范围配置 | `D:\MyProject\college\ecg1\.gitignore` | 排除虚拟环境、缓存、本机配置、原始 ECG 和嵌套 Obsidian 仓库 |
| GitHub CLI | `D:\MyProject\college\ecg1\outputs\tools\gh` | 本机 GitHub 登录工具；已被 Git 忽略。当前 Git 凭据助手引用它，继续上传时保留 |
| CLI 下载包 | `D:\MyProject\college\ecg1\outputs\tools\gh.zip` | 已完成解压，可删除 |
| PTB-XL 原始数据 | `D:\datasets\ptb-xl` | 单独存储，不随仓库上传 |
| LUDB 原始数据 | `D:\datasets\ludb` | 单独存储，不随仓库上传 |

## 完成清单

- [x] 创建私有 GitHub 仓库并设置默认分支 main。
- [x] 整理仓库首页和小组运行入口。
- [x] 发布代码、正式模型、指标、图表、STM32 导出与 LUDB 验证文档。
- [x] 排除虚拟环境、本机配置、缓存、原始 ECG、工具下载和嵌套 Obsidian 仓库。
- [x] 核对本地 Markdown 链接；待上传文本的常见凭据格式检查未发现匹配。
- [x] 发布前全量测试：40 passed，1 条既有 Starlette/httpx 弃用警告。
- [x] 上传并核对远端 main 与本机提交一致。
- [ ] 添加小组成员访问权限：需要组员 GitHub 用户名，并由成员接受邀请。
