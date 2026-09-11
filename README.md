# 🎮 Delta Exit Assistant —— 多游戏结算检测托盘助手（V2）

一个用于**检测游戏结算画面**并提醒你“该下机了”的 Windows 托盘工具。  
V2 版本已升级为 **多游戏 Profile 架构**：每个游戏独立 ROI 与模板组，可在托盘中切换游戏，且支持一键抓取模板与在线微调 ROI。

---

## ✨ V2 核心特性

- ✅ **多游戏支持（GameProfile）**
  - 每个游戏一个 profile：`ROI（相对坐标） + 模板组`
  - 托盘菜单一键切换当前游戏

- ✅ **相对 ROI（适配分辨率/缩放）**
  - `roi_rel = {x,y,w,h}` 采用屏幕比例，不再写死像素
  - 支持 Windows DPI Awareness（125%/150% 等缩放更稳）

- ✅ **模板组匹配：Best Match**
  - 当前 profile 下只匹配它自己的 ROI + 模板集合
  - 不扫描多个区域，不匹配多个游戏，资源占用更可控

- ✅ **结算界面“只提示一次”**
  - 内置 **armed/hysteresis 回落机制 + cooldown 冷却**：
    - 停留在结算界面不会疯狂重复提醒
    - 退出结算（分数跌破回落阈值）才会重新武装

- ✅ **通知系统（Win11 Toast） + 提示音**
  - 使用 `winotify` 发送 Windows 通知（打包后稳定）
  - 支持：**只响铃 / 只弹窗 / 都要**

- ✅ **托盘 UI 可调参数**
  - threshold / hysteresis / cooldown / scan_interval
  - 文本模板可编辑（支持占位符）
  - “发送测试通知”一键验证

- ✅ **按目标时间延后提醒**
  - 可启用“计时提醒”，设置一个目标时间（例如 `08:00`）
  - 每个游戏各自配置“游戏大概时长”
  - 例如目标时间为 `08:00`、游戏时长为 `30` 分钟时，程序从 `07:30` 起才允许结算提醒
  - 窗口外仍会继续截图和匹配，但不会触发通知；窗口开启时，已显示的结算页可立即提醒

- ✅ **抓取模板（无需截图软件）**
  - 托盘菜单：`抓取模板 → 抓取：胜利/失败/平局...`
  - 自动截取当前游戏 ROI 并保存为对应模板图片
  - 保存位置：优先写入 exe 同级的 `assets/templates/...`（可写、可持续）

- ✅ **ROI 调整器（roi_tuner）**
  - 托盘菜单：`ROI 调整 → 预览 ROI / 移动 / 缩放 / 步长 / 恢复默认`
  - 调整结果存入 `config.json` 的 `roi_overrides`，不污染 profile.json
  - 适配不同玩家 UI/分辨率/字体渲染差异

---

## 🧠 工作原理（简述）

Detector 只关心一件事：

> 当前选中的 `GameProfile` → 截该 profile 的 ROI → 在该 profile 的模板组里取 best match → 达到阈值并满足回落/冷却 → 触发回调

托盘 UI 负责：
- 选择游戏 profile
- 调整参数/文案/提醒方式
- 抓取模板
- 微调 ROI

---

## 📁 项目结构（V2）
****
assets/
icon.ico
profiles/
delta.json
valorant.json
...
templates/
delta/
success.png
fail.png
valorant/
win.png
lose.png
draw.png

src/
app.py # 入口：只负责启动 TrayApp
ui_tray.py # 托盘 UI & 菜单动作（主逻辑）
ui_dialogs.py # Tk 对话框服务（线程安全）
notify.py # 通知/响铃/文本模板渲染
capture.py # 抓取 ROI 保存为模板、ROI 预览截图
roi_tuner.py # ROI 微调器（移动/缩放/步长/重置）
config_store.py # config.json 读写
detector.py # 检测核心（armed + hysteresis + cooldown）
profiles.py # profile 加载 + 资源路径解析（支持 runtime override）
roi.py # 相对 ROI → 像素 ROI


---

## 🧩 GameProfile 配置（assets/profiles/*.json）

每个游戏一个 profile，例如 `assets/profiles/valorant.json`：

```json
{
  "id": "valorant",
  "display_name": "无畏契约",
  "estimated_duration_min": 40,
  "roi_rel": { "x": 0.340104, "y": 0.301852, "w": 0.227083, "h": 0.400926 },
  "templates": [
    { "id": "valorant_win",  "label": "胜利", "path": "assets/templates/valorant/win.png" },
    { "id": "valorant_lose", "label": "败北", "path": "assets/templates/valorant/lose.png" },
    { "id": "valorant_draw", "label": "平局", "path": "assets/templates/valorant/draw.png" }
  ]
}
roi_rel 四个参数含义

x：ROI 左上角横坐标占屏幕宽度比例

y：ROI 左上角纵坐标占屏幕高度比例

w：ROI 宽度占屏幕宽度比例

h：ROI 高度占屏幕高度比例

🛠️ 使用指南（推荐流程）
1）选择游戏

托盘 → 选择游戏 → 选择对应 profile

2）校准 ROI（如果截图歪）

托盘 → ROI 调整

“预览 ROI”查看当前截取区域

用 “上/下/左/右”移动 ROI

用 “加宽/变窄/加高/变矮”调整大小

“设置步长”建议：

1920×1080 下：0.005 ≈ 9~10 像素

更精细：0.002

调整结果会写入 config.json（roi_overrides），不改 profile.json。

3）抓取模板

进入游戏结算界面（胜利/失败/平局）
托盘 → 抓取模板 → 选择对应项（例如“抓取：胜利”）

模板保存到：

assets/templates/<game>/<xxx>.png（优先写在 exe 同级，可持久）

4）启动检测

托盘 → 启动检测
满足阈值后弹窗/响铃提醒。

5）按时间控制提醒（可选）

托盘 → 设置 → 计时提醒：

- 勾选“启用计时提醒”
- 设置目标时间，例如 `08:00`
- 切换到对应游戏，设置该游戏的“游戏大概时长”

例如目标时间为 `08:00`，三角洲行动大概时长为 `30` 分钟，则 `07:30` 前即使匹配到结算界面也不会提醒；从 `07:30` 起恢复正常提醒。目标时间过去后，提醒继续保持开启，避免一局超时后错过结算提醒。

⚙️ 设置（托盘 → 设置）

可调参数：

threshold：匹配阈值（建议 0.70~0.90）

hysteresis：回落差值（建议 0.08~0.20）

cooldown_sec：冷却时间（秒）

scan_interval_sec：扫描间隔（秒，越小越灵敏但更耗资源）

提醒方式：

都要（弹窗 + 响铃）

只弹窗

只响铃

通知文本：

标题/正文都可编辑，支持占位符：

{game} 当前游戏名

{label} 模板 label（来自 profile.json）

{score} 匹配分数（支持 {score:.3f}）

{id} 模板 ID

📦 打包（PyInstaller）

建议使用：

python -m PyInstaller --clean -F -w --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --add-data "assets;assets"

生成位置：

dist/Delta-Exit-Assistant.exe

注意：V2 支持运行时写入模板（抓取模板）与 config.json，建议把 exe 放在你有写权限的目录（例如桌面/某个文件夹），避免放在系统受限目录。

🧯 常见问题
1）为什么停留在结算界面不会反复提醒？

V2 使用 “armed + hysteresis + cooldown”：

达到阈值触发一次后 armed=False

只有分数跌破 (threshold - hysteresis) 才重新 armed=True

2）为什么某个模板文件不存在会出问题？

建议先放占位图（复制 win/lose 任一张），或先用“抓取模板”生成对应文件。

3）模板截图偏了怎么办？

用 “ROI 调整 → 预览 ROI” 先把区域调准，再抓取模板。

✅ V2 版本里程碑

V2 到此为止，后续功能（如引导式抓取模板、用户自定义新增游戏、模板管理器、多屏选择等）可在此架构上继续扩展。
