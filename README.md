# 🎮 Delta Exit Assistant（下机助手）

[简体中文](README.md) | [日本語](README_ja.md) | [English](README_en.md)

一个 Windows 托盘常驻的“结算检测提醒器”：当你在游戏里打完一局、进入结算界面时，程序会自动识别结算画面（胜利/失败/平局等），并用 **Windows 通知弹窗 / 提示音**提醒你“该下机了”。

项目目标是做一个**轻量、低打扰、可扩展到多游戏**的结算检测工具。

---

## ✨ 功能特性

* ✅ **托盘常驻**：无窗口运行，右键菜单控制
* ✅ **多游戏 Profile**：每个游戏一套配置（ROI + 模板）
* ✅ **自定义游戏流程**：在托盘里新建游戏 → 框选 ROI → 抓取模板 → 立即可用
* ✅ **相对 ROI 坐标**：`roi_rel` 用 0~1 的相对比例，适配不同分辨率/缩放
* ✅ **模板匹配识别**：OpenCV `matchTemplate`，取 best match
* ✅ **冷却机制**：停留在结算界面不会疯狂重复提醒
* ✅ **睡觉提醒时间窗**：按全局睡觉时间和每游戏提前分钟数，在时间窗内每局结算都提醒
* ✅ **多语言 UI**：托盘菜单支持中文、日文和英文，可随时切换
* ✅ **提醒模式可选**：只响铃 / 只弹窗 / 都要
* ✅ **可配置提示文本**：支持 `{game}/{label}/{score}/{id}` 占位符
* ✅ **可手动微调 ROI**：托盘里提供移动/缩放步长微调 + 预览

---

## 🧠 核心实现思路（技术方法）

### 1) Profile 化：把“不同游戏”抽象成配置

每个游戏一个 `profile.json`，包含：

* `id`：游戏标识（用于文件名与模板目录）
* `display_name`：托盘显示名
* `sleep_lead_minutes`：该游戏在睡觉时间前提前多少分钟开始提醒，旧 Profile 默认使用 `30`
* `roi_rel`：截图区域（相对坐标）
* `templates[]`：模板列表（每个模板是一张图片，代表一种结算结果）

示例（`assets/profiles/valorant.json`）：

```json
{
  "id": "valorant",
  "display_name": "无畏契约",
  "sleep_lead_minutes": 45,
  "roi_rel": { "x": 0.34, "y": 0.30, "w": 0.227, "h": 0.401 },
  "templates": [
    { "id": "valorant_win",  "label": "胜利", "path": "assets/templates/valorant/win.png" },
    { "id": "valorant_lose", "label": "败北", "path": "assets/templates/valorant/lose.png" },
    { "id": "valorant_draw", "label": "平局", "path": "assets/templates/valorant/draw.png" }
  ]
}
```

---

### 2) 睡觉提醒时间窗

睡觉提醒默认开启。全局睡觉时间保存在运行目录的 `config.json`，每个游戏的提前提醒分钟数保存在对应 Profile 中。

例如睡觉时间为 `22:00`，当前无畏契约 Profile 的 `sleep_lead_minutes` 为 `45`，提醒窗口就是 `21:15 ~ 22:00`。在窗口内，每次 Detector 识别到新的结算画面都会发送睡觉提醒。

全局配置默认值：

```json
{
  "ui_language": "zh",
  "sleep_reminder_enabled": true,
  "sleep_bed_time": "22:00",
  "sleep_stop_after_bed_time": true,
  "sleep_auto_start_detection": true,
  "sleep_title_tpl": "{game} 睡觉提醒",
  "sleep_msg_tpl": "已经到休息时间了，这把结束后就下机。"
}
```

托盘 `设置 → 睡觉提醒` 中可以修改：

* 是否启用睡觉提醒
* 全局睡觉时间
* 当前游戏的提前提醒分钟数
* 到达睡觉时间后是否停止睡觉提醒
* 进入提醒窗口后是否自动启动检测
* 睡觉提醒标题和正文

界面语言保存在 `config.json` 的 `ui_language` 字段中，可选值为 `zh`、`ja`、`en`。

---

### 3) ROI（截图区域）用“相对坐标”表示

`roi_rel` 的四个参数含义：

* `x`：ROI 左上角相对屏幕宽度的位置（0~1）
* `y`：ROI 左上角相对屏幕高度的位置（0~1）
* `w`：ROI 宽度相对屏幕宽度比例（0~1）
* `h`：ROI 高度相对屏幕高度比例（0~1）

运行时会用 `screen_w/screen_h` 把它转成像素坐标 `RoiPx(left, top, width, height)`。

这样能适配：

* 不同分辨率（1080p/2K/4K）
* 不同缩放（100%/125%/150%）

---

### 4) 检测链路（Detector）

检测器的职责只做一件事：

> 当前选中的 `GameProfile` → 截取该 profile 的 ROI → 在模板里取 best match → 达到阈值后触发回调

流程：

1. 用 `mss` 截屏 ROI
2. ROI 转灰度
3. 对每个模板做 `cv2.matchTemplate`
4. 取最大 `score` 的模板作为 best match
5. `score >= threshold` 且满足冷却条件 → 触发提醒

> 注意：模板文件允许不存在（例如 “平局”还没抓取），检测器会跳过缺失模板，不影响其它模板工作。

---

### 5) 冷却机制（避免重复弹窗/响铃）

当你停留在结算界面时，识别会一直命中同一个模板。

为避免反复提醒，Detector 使用：

* `cooldown_sec`：触发一次提醒后，至少隔 N 秒才允许再次触发
* `hysteresis`：触发后需要 score 下降到 `threshold - hysteresis` 以下才“重新武装”

两者结合可以做到：

* 结算界面停留不刷屏
* 真正离开结算界面后再进入，会再次提醒

---

### 6) 托盘 UI（pystray）

托盘提供的关键入口：

* **启动检测 / 停止检测**
* **选择游戏（profiles 列表）**
* **新建游戏…**
* **抓取模板（胜利/败北/平局…）**
* **ROI 调整**（移动/缩放/预览/恢复默认）
* **设置**（阈值、冷却、扫描间隔、提醒方式、文本占位符）
* **退出**

---

### 7) 新建游戏完整工作流（Custom Game）

这是本项目“可扩展到任意游戏”的核心能力：

1. 托盘 → `新建游戏…`
2. 输入游戏名（例如 `ow2`）
3. 弹出全屏遮罩，鼠标拖拽框选 ROI
4. 自动创建：

   * `assets/profiles/ow2.json`
   * `assets/templates/ow2/`（空目录）
5. 自动切换到 ow2 profile
6. 在游戏结算界面 → `抓取模板` → 抓取胜利/失败/平局
7. 下次启动软件会遍历 `assets/profiles`，ow2 会永久存在并可用

---

## 📦 文件结构

运行目录结构（推荐）：

```text
📦 程序根目录
 ┣ 📜 Delta-Exit-Assistant.exe
 ┗ 📂 assets
    ┣ 📂 profiles
    ┃  ┣ 📜 delta.json
    ┃  ┣ 📜 valorant.json
    ┃  ┗ 📜 ow2.json          （新建游戏时生成）
    ┗ 📂 templates
       ┣ 📂 delta
       ┃  ┣ 🖼 success.png
       ┃  ┗ 🖼 fail.png
       ┣ 📂 valorant
       ┃  ┣ 🖼 win.png
       ┃  ┣ 🖼 lose.png
       ┃  ┗ 🖼 draw.png（可选）
       ┗ 📂 ow2
          ┣ 🖼 win.png        （抓取模板生成）
          ┣ 🖼 lose.png
          ┗ 🖼 draw.png（可选）
```

---

## 🚀 使用方法（给用户）

### 1) 运行

双击 `Delta-Exit-Assistant.exe`
程序会出现在系统托盘（右下角）。

### 2) 选择游戏并启动检测

托盘右键：

1. `选择游戏` → 选一个已有游戏
2. 点击 `启动检测`
3. 打完一局进入结算界面 → 识别成功会弹窗/响铃

### 3) 抓取模板（让识别更准）

进入“结算界面”时：

托盘右键 → `抓取模板` → 选择某个结果（胜利/败北/平局）
程序会把当前 ROI 截图保存为模板（png），并立即生效。

### 4) 新建游戏（适配其它游戏）

托盘右键 → `新建游戏…`

* 输入名称（例如 `ow2`）
* 框选 ROI（结算标题/关键字区域）
* 进入结算界面后抓取模板
* 下次启动即可直接选择该游戏

### 5) ROI 调整（截图歪了就用它）

如果识别不稳定，多数是 ROI 没框准。

托盘右键 → `ROI 调整`：

* `预览 ROI`：打开当前截图看看是否框对
* 用方向键菜单移动 ROI
* 用加宽/变窄/加高/变矮微调
* 需要时可恢复默认 ROI

### 6) 设置提醒方式与文案

托盘右键 → `设置`：

* **提醒方式**：都要 / 只弹窗 / 只响铃
* **通知文本**：标题和正文支持占位符：

占位符：

* `{game}` 当前游戏名
* `{label}` 结果标签（来自 profile 的 templates[].label）
* `{score}` 匹配分数（可用 `{score:.3f}` 格式）
* `{id}` 模板 ID

### 7) 切换界面语言

托盘右键 → `设置` → `界面语言`，可以选择：

* `中文`
* `日本語`
* `English`

切换后托盘菜单和后续弹窗会立即更新。用户自行编辑过的通知文案会保留不变。

---

## 🛠️ 开发 & 安装依赖

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

---

## 🧱 打包（记录）

本项目采用 **onedir 外置 assets**（便于动态写入模板/新建 profiles）。

在项目根目录运行：

```powershell
python -m PyInstaller --clean --noconfirm -w --onedir --contents-directory "." --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --icon "assets/icon.ico" --add-data "assets;assets"
```

打包结果在：

```text
dist/Delta-Exit-Assistant/
  Delta-Exit-Assistant.exe
  assets/...
```

> 如果你后续使用 Inno Setup 打包安装器，把整个 `dist/Delta-Exit-Assistant` 作为安装目录即可。
> 当前发布方式建议保持 ZIP 解压即用，并解压到普通可写目录，以便保存 `config.json`、Profile 和模板图片。

如果需要一个更直观的单文件启动入口，可额外构建 onefile 版本：

```powershell
python -m PyInstaller --clean --noconfirm -w --onefile --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --icon "assets/icon.ico"
```

将生成的 `Delta-Exit-Assistant.exe` 与外置 `assets/` 放在同一目录。用户只需双击 exe；`assets/` 仍需保留，以便读取 Profile 并保存自定义模板。

---

## 📋 开发计划

* 支持多显示器，并允许选择游戏所在屏幕。当前版本暂时只按主屏幕截图。

---

## ⚠️ 常见问题

### Q1：新建游戏后“抓取模板失败：FileNotFoundError”

通常是因为模板路径/目录不在 exe 同目录下的 `assets/`。
请确认运行目录结构如上，且 exe 同级有 `assets/templates/<game>/`。

### Q2：识别一直不触发/误触发

* 先用 `ROI 调整 → 预览 ROI`，确认框到的就是“结算关键文字区域”
* 提高阈值 `threshold` 会减少误报
* 增大 `cooldown_sec` 会减少重复提醒
* 如果字体/背景变化大，建议抓取更“稳定”的结算区域作为模板

---

## 📌 免责声明

本项目仅基于 **本机屏幕截图 + 图像模板匹配**进行提醒，不注入游戏、不读写游戏内存、不修改游戏文件。请合理使用，避免影响正常游戏体验。
使用本项目造成的任何后果（包括但不限于误报/漏报、通知异常、与反作弊规则冲突等）由使用者自行承担。
请在遵守游戏服务条款与当地法律法规的前提下使用。

## 🌹 作者的话

这个项目的出发点很简单：我经常会想着“打完这把就下”，但结算页一过又继续排队。
于是我做了一个托盘常驻的小工具：只通过屏幕截图 + 模板匹配识别结算界面，在关键时刻弹窗/响铃提醒我停手。

它不是外挂，不注入进程、不读写内存、不修改游戏文件；只是一个“自控提醒器”。
我把它开源出来，希望你也能更轻松地管理自己的游戏时间，也欢迎你为更多游戏添加 Profile 或改进本项目。
未来我也许会和我的好友xjj一起发布更倾向于“自控软件”的版本，至于目前，先到这了。

如果你基于本项目二改或发布衍生版本，请保留作者署名与 LICENSE。谢谢。

## ⚠️ License: MIT
本项目允许自由使用、修改和分发（包括商业用途），但需保留原作者署名与许可文本。
本软件按 “原样” 提供，不对任何损失承担责任。
