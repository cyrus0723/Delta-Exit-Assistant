# 🎮 Delta Exit Assistant----三角洲下机助手  
一个用于《三角洲行动》结算检测的 Windows 托盘提醒工具。

每次都心里暗暗想"打完这把就下"，却总是会上头后一把接着一把？
想定闹钟却发现打游戏的时候响铃很吵，而关闭闹钟后却总是会被紧张的游戏搞忘记？
三角洲下机助手来辣！本工具正是为了解决这些问题而生。
当检测到“撤离成功”或“撤离失败”结算页面时，会自动弹出系统通知并发出提示音，帮助你控制游戏局数。

---

## ✨ 功能特性

- 🖥 后台常驻（系统托盘运行）
- 🔔 Windows 原生 Toast 通知（Win11）
- 🔊 声音提示兜底（即使全屏模式也可听到）
- 🎯 模板匹配检测结算页面
- ⚙️ 托盘菜单支持：
  - 两者都提醒
  - 只提醒成功
  - 只提醒失败
- 💾 自动保存配置（config.json）
- 📦 支持打包为单文件 exe（无需 Python 环境）

---

## 📂 项目结构
game_controler/
├─ src/
│ ├─ app.py
│ └─ detector.py
├─ assets/
│ ├─ icon.ico
│ └─ templates/
│  ├─ success.png
│  └─ fail.png
├─ requirements.txt
└─ README.md

---

## 🧠 工作原理

- 使用 `mss` 截取屏幕指定 ROI 区域
- 使用 `OpenCV` 模板匹配识别结算页面
- 达到阈值后触发提醒（带冷却机制防止重复触发）

---

## 🖼 模板准备

在 `assets/templates/` 中放入：

- `success.png`（包含“撤离成功”文字区域）
- `fail.png`（包含“撤离失败”文字区域）

建议：
- 仅截取标题区域
- 尽量裁剪干净，避免多余背景
- 使用与你游戏分辨率一致的截图

---

## 🛠 开发环境运行

### 1️⃣ 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2️⃣ 安装依赖
pip install -r requirements.txt

3️⃣ 运行程序
python .\src\app.py

托盘图标出现后即可使用。

📦 打包为 exe（无需 Python 环境）

在项目根目录执行：

pip install pyinstaller pyinstaller-hooks-contrib
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "DeltaResultTray" ^
  --icon ".\assets\icon.ico" ^
  --add-data ".\assets;assets" ^
  --paths ".\src" ^
  --collect-all winotify ^
  .\src\app.py

生成文件：

dist/DeltaResultTray.exe

该 exe 可直接复制到其他 Windows 电脑运行。
⚙️ 配置文件

程序运行后会在 exe 同目录生成：

config.json

可修改参数：

{
  "threshold": 0.82,
  "cooldown_sec": 8.0,
  "mode": "both"
}

参数说明：

参数	            说明
threshold	        模板匹配阈值
cooldown_sec	    提醒冷却时间（秒）
mode	            both / success / fail
⚠️ 注意事项

建议使用 无边框窗口模式 运行游戏
全屏独占模式可能导致部分机器无法截图
若通知被系统“专注助手”压制，仍会发出声音提示
若分辨率改变，需要重新采集模板

🙌 免责声明
本项目仅用于学习与个人效率管理用途。
请勿用于违反游戏规则的行为。

未来升级方向：
·适配各种分辨率，屏幕缩放比
·适配各种游戏的结算界面
·自定义提示音和提示形式
·自定义局数（按结算次数提醒）
