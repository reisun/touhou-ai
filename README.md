# Touhou AI

東方風神録（Touhou 10）の実機プレイを使った、PPO強化学習の実験プロジェクトです。
ゲーム内の状態から観測を作り、移動・低速移動・ショット・ボムを学習します。
実機連携、回避シミュレーター、学習状況を表示するダッシュボードを含みます。

研究・開発途中です。クリア性能や、すべてのゲーム環境での動作を保証するものではありません。

## 主な構成

- `touhou_ai/`：観測生成、小型CNN、PPO、報酬、実機連携
- `scripts/`：起動・停止、検証、実験用スクリプト
- `dashboard/`：学習状況とOBS向け表示
- `tests/`：単体テスト・観測や当たり判定の検証
- `docs/`：設計、実験結果、変更履歴

全体グリッドと自機周辺グリッド、方向別の短期衝突予測を入力します。
実機操作は基本2F単位です。通常の報酬は射撃ダメージ・進行度・被弾・Power取得を対象とし、
係数や実験条件は変更されるため、[報酬実装](touhou_ai/live_rewards.py)を参照してください。

## 必要な環境

- Windows、PowerShell 7、Python 3.11以降
- 正規に入手した東方風神録。実機連携は検証対象の実行ファイルに限定されます
- Python依存関係：`requirements-learner.txt`、`requirements-windows-live.txt`
- DirectMLを使う場合は対応するPyTorch環境。Docker / WSLはモック実験向けの任意構成です

## 初期セットアップ

```powershell
./scripts/setup.ps1 -Python C:\path\to\python.exe
./.venv/Scripts/python.exe -m pip install -r requirements-learner.txt -r requirements-windows-live.txt
Copy-Item game.example.json game.local.json
```

`game.local.json`に自分の実行ファイルのパスなどを設定します。
実行ファイルの検証を通過する必要があり、ハッシュを書き換えるだけで別バージョンに対応できるわけではありません。

実機の準備・制約は[Windows probe](docs/windows-probe.md)、
学習運用は[実機学習](docs/real-learning.md)と[起動スクリプト](scripts/live-learning.ps1)を参照してください。
過去の文書には当時の実験条件が記載されており、現在の実装と異なる場合があります。

実行中の学習を確認・停止する例：

```powershell
./scripts/live-learning.ps1 status
./scripts/live-learning.ps1 stop
```

表示サーバーは`./scripts/dashboard.ps1 start`で起動します。
実験ログ・保存モデルは`artifacts/`、実行時設定は`.runtime/`へ保存されます。
これらと`.env`、`game.local.json`はGit管理対象外です。
初期基盤のモックは実機学習とは別で、`bridge.py`とDocker構成に残っています。

## 検証と記録

```powershell
./.venv/Scripts/python.exe -m unittest tests.test_live_rewards tests.test_dual_grid
```

- [モデル・学習の変更履歴](docs/model-learning-history.md)
- [グリッド観測の設計](docs/dual-grid-design.md)
- [実機連携と調査元](docs/live-adapter.md)

文書内のローカル実験成果物や保存モデルは公開リポジトリには含まれません。
実機用ツールはゲームへの入力・状態取得を行います。ダッシュボードや操作用サーバーは
ローカル利用を想定しており、インターネットへ公開しないでください。

## ライセンス

本プロジェクトの独自部分は[MIT License](LICENSE)です。改変・再配布・商用利用が可能で、
著作権表示とライセンス文の保持が必要です。
第三者由来部分の表記は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。

東方Projectおよびゲーム・素材の権利はそれぞれの権利者に帰属します。
ゲーム本体や素材は配布していません。本プロジェクトは非公式です。
