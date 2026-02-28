import time
from pathlib import Path

import cv2
import numpy as np
from mss import mss
from plyer import notification

# Windows 自带：用于提示音（不额外依赖）
import winsound

# =========================
# 固定配置（按你的情况）
# =========================
SCREEN_W, SCREEN_H = 1920, 1080

# 结算标题所在区域（只截这里，速度快、误报低）
ROI = {"left": 150, "top": 150, "width": 500, "height": 160}

# 模板文件
TEMPL_DIR = Path("templates")
T_WIN = TEMPL_DIR / "success.png"
T_LOSE = TEMPL_DIR / "fail.png"

# 匹配阈值：0.78~0.90 之间调；先用 0.82
THRESHOLD = 0.82

# “回滞”阈值：用于判断“已离开结算页”，防止临界抖动导致反复进出
# 例如 THRESHOLD=0.82，则 best < 0.74 才认为离开（0.08 可调）
HYSTERESIS = 0.08

# 扫描频率
SCAN_INTERVAL = 0.20

# 提示音：True=开启；想静音就改 False
ENABLE_BEEP = True

# 提示音类型（可选 MB_OK / MB_ICONASTERISK / MB_ICONEXCLAMATION / MB_ICONHAND）
BEEP_TYPE = winsound.MB_ICONASTERISK

# 每次启动时提醒你“以后可选换 winotify”（只提示一次）
PRINT_WINOTIFY_HINT_ON_START = True
# =========================


def grab_roi(sct: mss, roi: dict) -> np.ndarray:
    """抓取 ROI，返回 BGR 图像"""
    shot = sct.grab(roi)
    frame = np.array(shot)[:, :, :3]  # BGRA->BGR
    return frame


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """轻量预处理：灰度 + 轻微去噪"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"读不到模板：{path}")
    return img


def match_score(screen_gray: np.ndarray, templ_gray: np.ndarray) -> float:
    res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


def capture_template(name: str, out_path: Path):
    """在结算画面时运行此函数，采集 ROI 保存为模板"""
    TEMPL_DIR.mkdir(parents=True, exist_ok=True)
    with mss() as sct:
        frame = grab_roi(sct, ROI)
    cv2.imwrite(str(out_path), frame)
    print(
        f"[OK] 已保存 {name} 模板到：{out_path}\n"
        f"提示：请确保此时画面正处于{name}结算页，且ROI内包含“撤离成功/撤离失败”等固定文字。"
    )


def notify(result: str, score: float):
    """通知 + 提示音（尽量不依赖横幅展示）"""
    # 通知（可能被全屏/请勿打扰压横幅，但会进通知中心）
    notification.notify(
        title="三角洲行动：检测到结算",
        message=f"{result}（匹配度 {score:.2f}）\n打完这把就停手？",
        timeout=5
    )

    # 声音兜底：横幅不出也能听到
    if ENABLE_BEEP:
        try:
            winsound.MessageBeep(BEEP_TYPE)
        except Exception:
            # 极少数情况下音频设备/策略异常，忽略即可
            pass


def watch():
    templ_win = load_gray(T_WIN)
    templ_lose = load_gray(T_LOSE)

    # “边沿触发”状态：在结算页时 True，离开结算页才会恢复 False
    in_result_screen = False

    if PRINT_WINOTIFY_HINT_ON_START:
        print("提示：如果你以后想要更原生、更稳定的 Win11 Toast，可以考虑把通知库换成 winotify（当前 plyer 已能用）。\n")

    print("开始监测结算标题区域（ROI）...")
    print(f"ROI = {ROI}")
    print("按 Ctrl+C 退出。\n")

    with mss() as sct:
        while True:
            frame = grab_roi(sct, ROI)
            gray = preprocess(frame)

            s_win = match_score(gray, templ_win)
            s_lose = match_score(gray, templ_lose)

            best = max(s_win, s_lose)

            # 进入结算：只触发一次
            if best >= THRESHOLD and not in_result_screen:
                result = "撤离成功" if s_win >= s_lose else "撤离失败"
                notify(result, best)
                print(f"[触发] {result} score={best:.2f}")
                in_result_screen = True

            # 离开结算：允许下一次触发（用回滞避免临界抖动）
            elif in_result_screen and best < (THRESHOLD - HYSTERESIS):
                in_result_screen = False

            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["success", "fail"], help="采集模板：在对应结算页运行")
    args = parser.parse_args()

    if args.capture == "success":
        capture_template("撤离成功", T_WIN)
    elif args.capture == "fail":
        capture_template("撤离失败", T_LOSE)
    else:
        if not T_WIN.exists() or not T_LOSE.exists():
            print("缺少模板！请先采集：")
            print("  python delta_extract_watch.py --capture success   （在撤离成功结算页执行）")
            print("  python delta_extract_watch.py --capture fail      （在撤离失败结算页执行）")
            raise SystemExit(1)
        watch()