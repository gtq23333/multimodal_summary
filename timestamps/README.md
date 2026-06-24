# Git 第三方时间锚（OpenTimestamps）

本目录存放 **OpenTimestamps (OTS)** 证明，用于将 Git commit 的密码学摘要锚定到 Bitcoin 公开账本。任何人可独立验证，**不依赖本机时钟**。

## 这是什么 / 不是什么

| 类型 | 本方案 | 法定「可信时间戳」(RFC 3161 / 国内 TSA) |
|------|--------|----------------------------------------|
| 是否需要你注册账号 | **否** | **是**（实名 + 付费/签约） |
| 能否防本机改时间 | **能**（锚定到 Bitcoin 区块时间） | **能**（TSA 签发令牌） |
| 司法/专利场景认可度 | 技术可验证，**不等同**于法定时间戳 | 在合规 TSA 下**可能**具备更强效力 |
| 当前仓库是否已配置 | **是**（见 `commits/*.commit.ots`） | **否**（需你自行到存证平台操作） |

## 生成时间戳

```bash
pip install opentimestamps-client   # 若尚未安装
python scripts/stamp_git_opentimestamps.py --all-commits
```

仅对最新 commit：

```bash
python scripts/stamp_git_opentimestamps.py
```

Bitcoin 出块后升级 pending 证明（约 10–60 分钟）：

```bash
python scripts/stamp_git_opentimestamps.py --upgrade
```

## 验证

1. 确认 `commits/<hash>.commit` 内容与当时 commit 一致（含 tree hash）。
2. 在 Linux/macOS 或 WSL 中安装 `opentimestamps-client`，运行：

   ```bash
   ots verify commits/<hash>.commit.ots
   ```

   Windows 原生 `ots` CLI 可能因 OpenSSL 依赖报错，可用 WSL/Docker，或使用本仓库脚本重新 `--upgrade` 后在线验证。

3. 验证成功时，证明会给出 **不晚于** 某 Bitcoin 区块时间的密码学下界。

## 若需要法定效力

需你本人到国内 **可信时间戳服务中心 / 公证存证 / 区块链存证平台**（如各 CA 时间戳服务、公证处电子存证等）完成实名认证并上传文件哈希。完成后把 **存证编号 / 时间戳令牌 / 证书 PDF** 放入 `timestamps/legal/`（需自行创建），我无法代你完成实名与付费环节。

## 文件说明

- `manifest.txt` — 本次批量锚定记录
- `commits/<short-hash>.commit` — 可公开的 commit 摘要（含 tree hash，不含源码）
- `commits/<short-hash>.commit.ots` — OpenTimestamps 证明（应纳入 Git 一并推送）
