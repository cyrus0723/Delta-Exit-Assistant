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


def notify(result: str, score: float, extra: str = ""):
    """通知 + 提示音（尽量不依赖横幅展示）"""
    msg = f"{result}（匹配度 {score:.2f}）\n打完这把就停手？"
    if extra:
        msg += f"\n{extra}"

    # 通知（可能被全屏/请勿打扰压横幅，但会进通知中心）
    notification.notify(
        title="三角洲行动：检测到结算",
        message=msg,
        timeout=5
    )

    # 声音兜底：横幅不出也能听到
    if ENABLE_BEEP:
        try:
            winsound.MessageBeep(BEEP_TYPE)
        except Exception:
            pass


def _should_fire_total(total_count: int, every: int) -> bool:
    return every > 0 and (total_count % every == 0)


def _should_fire_each(win_count: int, lose_count: int, every_win: int, every_lose: int, result: str) -> bool:
    if result == "撤离成功":
        return every_win > 0 and (win_count % every_win == 0)
    else:
        return every_lose > 0 and (lose_count % every_lose == 0)


def watch(every: int,
          mode: str,
          start_count: int,
          every_win: int,
          every_lose: int,
          cooldown: float):
    """
    every: 通用 N（mode=total/success/fail 时用）
    mode: total / success / fail / each
    start_count: 初始计数（total/success/fail 各自口径在下面会分别初始化）
    every_win/every_lose: mode=each 时分别使用
    cooldown: 触发提醒后的冷却秒数
    """
    templ_win = load_gray(T_WIN)
    templ_lose = load_gray(T_LOSE)

    # “边沿触发”状态：在结算页时 True，离开结算页才会恢复 False
    in_result_screen = False

    # 冷却：避免极端情况下（比如ROI抖动）短时间内反复提醒
    last_notify_ts = 0.0

    # 计数器
    total_count = start_count if mode == "total" else 0
    win_count = start_count if mode == "success" else 0
    lose_count = start_count if mode == "fail" else 0

    # mode=each 时，start_count 同时作用到两条计数更直观：可自行改成分别传参
    if mode == "each":
        win_count = start_count
        lose_count = start_count
        total_count = 0

    if PRINT_WINOTIFY_HINT_ON_START:
        print("提示：如果你以后想要更原生、更稳定的 Win11 Toast，可以考虑把通知库换成 winotify（当前 plyer 已能用）。\n")

    print("开始监测结算标题区域（ROI）...")
    print(f"ROI = {ROI}")
    print(f"THRESHOLD={THRESHOLD:.2f}, HYSTERESIS={HYSTERESIS:.2f}, interval={SCAN_INTERVAL:.2f}s")
    print(f"mode={mode}")
    if mode == "each":
        print(f"提醒策略：成功每 {every_win} 次提醒一次；失败每 {every_lose} 次提醒一次；起始计数={start_count}")
    else:
        print(f"提醒策略：每 {every} 次（按 mode 口径计数）提醒一次；起始计数={start_count}")
    if cooldown > 0:
        print(f"cooldown={cooldown:.1f}s")
    print("按 Ctrl+C 退出。\n")

    with mss() as sct:
        while True:
            frame = grab_roi(sct, ROI)
            gray = preprocess(frame)

            s_win = match_score(gray, templ_win)
            s_lose = match_score(gray, templ_lose)
            best = max(s_win, s_lose)

            # 进入结算：只触发一次（边沿触发）
            if best >= THRESHOLD and not in_result_screen:
                result = "撤离成功" if s_win >= s_lose else "撤离失败"

                now = time.time()
                can_notify = (cooldown <= 0) or ((now - last_notify_ts) >= cooldown)

                fired = False
                extra_line = ""

                if mode == "total":
                    total_count += 1
                    fired = can_notify and _should_fire_total(total_count, every)
                    extra_line = f"总局数：{total_count}（每 {every} 局提醒）"

                elif mode == "success":
                    if result == "撤离成功":
                        win_count += 1
                        fired = can_notify and _should_fire_total(win_count, every)
                    extra_line = f"成功局数：{win_count}（每 {every} 次成功提醒）"

                elif mode == "fail":
                    if result == "撤离失败":
                        lose_count += 1
                        fired = can_notify and _should_fire_total(lose_count, every)
                    extra_line = f"失败局数：{lose_count}（每 {every} 次失败提醒）"

                elif mode == "each":
                    if result == "撤离成功":
                        win_count += 1
                    else:
                        lose_count += 1
                    fired = can_notify and _should_fire_each(win_count, lose_count, every_win, every_lose, result)
                    extra_line = f"成功：{win_count}（每 {every_win}） | 失败：{lose_count}（每 {every_lose}）"

                else:
                    # 防御：不该到这
                    extra_line = "（未知 mode）"

                # 控制台日志：每次检测到结算都打印计数
                print(f"[结算] {result} score={best:.2f} | {extra_line}")

                # 达到第 N 次才提醒
                if fired:
                    notify(result, best, extra=extra_line)
                    last_notify_ts = now
                    print("[提醒] 已弹出通知\n")
                else:
                    # 没提醒也给个清晰的提示
                    if not can_notify:
                        print("[跳过] 冷却中，未提醒\n")
                    else:
                        print("[跳过] 未达到提醒次数\n")

                in_result_screen = True

            # 离开结算：允许下一次触发（用回滞避免临界抖动）
            elif in_result_screen and best < (THRESHOLD - HYSTERESIS):
                in_result_screen = False

            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Delta Extract result watcher (template matching).")

    # 采集模板
    parser.add_argument("--capture", choices=["success", "fail"], help="采集模板：在对应结算页运行")

    # 局数控制：核心参数
    parser.add_argument("--every", type=int, default=1,
                        help="每 N 次（按 mode 口径）才提醒一次；mode=total/success/fail 时生效。默认 1=每次都提醒")
    parser.add_argument("--mode", choices=["total", "success", "fail", "each"], default="total",
                        help="计数口径：total=总局数；success=只数成功；fail=只数失败；each=成功/失败分别计数")

    # mode=each 时的独立 N
    parser.add_argument("--every-win", type=int, default=1, help="mode=each 时：成功每 N 次提醒一次")
    parser.add_argument("--every-lose", type=int, default=1, help="mode=each 时：失败每 N 次提醒一次")

    # 从中途开始计数（重启不断档）
    parser.add_argument("--start-count", type=int, default=0,
                        help="起始计数（从第几次开始数）。例如你已经打了 7 局，想让下一局算第 8 局，则填 7")

    # 提醒冷却
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="提醒后冷却秒数（防抖）。默认 0 不启用")

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

        # 参数校验
        if args.mode != "each":
            if args.every <= 0:
                raise SystemExit("--every 必须是正整数")
        else:
            if args.every_win <= 0 or args.every_lose <= 0:
                raise SystemExit("--every-win / --every-lose 必须是正整数")

        watch(
            every=args.every,
            mode=args.mode,
            start_count=args.start_count,
            every_win=args.every_win,
            every_lose=args.every_lose,
            cooldown=args.cooldown
        )