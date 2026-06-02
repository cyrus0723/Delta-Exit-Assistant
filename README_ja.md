# 🎮 Delta Exit Assistant（終了リマインダー）

[简体中文](README.md) | [日本語](README_ja.md) | [English](README_en.md)

Delta Exit Assistant は Windows のトレイ常駐アプリです。ゲームのリザルト画面を検出し、プレイを終了するタイミングを通知します。使用するのはローカルのスクリーンショットと OpenCV のテンプレートマッチングのみです。ゲームへの注入、メモリ読込、ファイル変更、入力自動化は行いません。

## 機能

* トレイメニューから検出の開始、停止、終了
* 相対 ROI 座標と結果テンプレートを持つゲーム Profile
* Delta Force と VALORANT の内蔵 Profile
* ゲーム追加、ROI 選択、テンプレート取得によるカスタムゲーム対応
* 同じ画面での連続通知を防ぐクールダウンとヒステリシス
* 通知方法：トースト、サウンド、または両方
* 全体の就寝時刻とゲーム別の事前通知時間による就寝リマインダー
* UI 言語：中国語、日本語、英語

## 就寝リマインダーの時間枠

就寝リマインダーは初期状態で有効です。全体の就寝時刻は `config.json` に保存され、ゲームごとの事前通知時間は Profile の `sleep_lead_minutes` に保存されます。

例えば就寝時刻が `22:00`、選択中の VALORANT Profile が `45` 分前の場合、通知時間枠は `21:15 ~ 22:00` です。この時間枠では、新しいリザルト画面を検出するたびに就寝通知が表示されます。

初期設定：

```json
{
  "ui_language": "ja",
  "sleep_reminder_enabled": true,
  "sleep_bed_time": "22:00",
  "sleep_stop_after_bed_time": true,
  "sleep_auto_start_detection": true,
  "sleep_title_tpl": "{game} 就寝リマインダー",
  "sleep_msg_tpl": "休む時間です。この試合が終わったらゲームを終了しましょう。"
}
```

`設定 → 就寝リマインダー` から就寝時刻、現在のゲームの事前通知時間、自動検出開始、通知テキストを変更できます。

## UI 言語の変更

トレイアイコンを右クリックし、`設定 → 表示言語` を開きます。`中文`、`日本語`、`English` から選択できます。トレイメニューと以降のダイアログはすぐに切り替わります。ユーザーが編集した通知テキストは保持されます。

## Profile の形式

各ゲームは `assets/profiles/<game>.json` を使用します。

```json
{
  "id": "valorant",
  "display_name": "VALORANT",
  "sleep_lead_minutes": 45,
  "roi_rel": { "x": 0.34, "y": 0.30, "w": 0.327, "h": 0.401 },
  "templates": [
    { "id": "valorant_win", "label": "勝利", "path": "assets/templates/valorant/win.png" },
    { "id": "valorant_lose", "label": "敗北", "path": "assets/templates/valorant/lose.png" }
  ]
}
```

`sleep_lead_minutes` がない古い Profile では `30` 分が使用されます。

## 使い方

1. `Delta-Exit-Assistant.exe` を起動します。
2. トレイアイコンを右クリックし、ゲームを選択します。
3. `検出を開始` をクリックします。
4. リザルト画面が一致すると、設定に応じて通知、サウンド、または両方が実行されます。

カスタムゲームでは `ゲームを追加…` を選び、リザルト画面の ROI をドラッグで選択した後、トレイメニューから勝利、敗北、必要に応じて引き分けのテンプレートを取得します。

## 開発

Python 3.10 以上を推奨します。

```bash
pip install -r requirements.txt
```

書込可能な onedir ZIP 配布としてパッケージ化します。

```powershell
python -m PyInstaller --clean --noconfirm -w --onedir --contents-directory "." --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --icon "assets/icon.ico" --add-data "assets;assets"
```

`config.json`、Profile、テンプレート画像を保存できるよう、通常の書込可能なフォルダーに展開してください。

単一ファイルの起動入口が必要な場合は、追加で onefile 版をビルドします。

```powershell
python -m PyInstaller --clean --noconfirm -w --onefile --paths "src" `
  "src/app.py" --name "Delta-Exit-Assistant" --icon "assets/icon.ico"
```

生成された `Delta-Exit-Assistant.exe` と外部 `assets/` フォルダーを同じディレクトリに置きます。ユーザーは exe をダブルクリックするだけで起動できます。Profile の読込とカスタムテンプレート保存のため、`assets/` は保持してください。

## 開発予定

* ゲームを表示するモニターを明示的に選択できるマルチモニター対応。現在のバージョンはプライマリモニターのみを取得します。

## ライセンス

MIT。派生版を再配布する場合は、元の著作者表示とライセンステキストを保持してください。
