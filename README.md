# Neural Network Visualizer

## 概要 (Overview)

ニューラルネットワークの動作原理を**組織・面接のメタファー**で理解できるインタラクティブWebアプリケーションです。

面接システムを例に、順伝播(Forward Propagation)と誤差逆伝播(Backpropagation)を直感的に学べます。

🌐 **Live Demo**: https://ofcbj.github.io/nn-book/

## ✨ 主な機能

### 🎯 インタラクティブ学習
- **リアルタイム可視化**: 各ニューロンの重み・バイアス・活性化値をアニメーション表示
- **ステップバイステップ**: 順伝播・逆伝播の各計算ステージを詳細に表示
- **手動モード**: 「次のステップ」ボタンでゆっくり学習過程を確認

### 🧠 ネットワーク構造
```
入力層 (3)     → 1次面接官 (5)  → 2次面接官 (3)  → 最終決定 (3)
[成績, 態度, 応答]  [Hidden Layer 1]  [Hidden Layer 2]  [不合格/保留/合格]
```

### 📊 詳細な可視化
- **Forward Pass**: 計算過程をポップアップで表示
  - 入力との内積計算
  - バイアス加算
  - Sigmoid活性化関数の適用
  
- **Backward Pass**: 誤差逆伝播を詳細に可視化
  - エラーの逆伝播
  - 勾配の計算
  - 重み・バイアスの更新量
  - 実際の更新アニメーション

- **Summary Modal**: 学習完了後に全体サマリーを表示
  - 更新された全重み・バイアスの変化量
  - Cross-Entropy Lossの計算過程

### 🌍 多言語対応
- **日本語** (Japanese)
- **한국어** (Korean)
- UIとアニメーションラベルが完全対応

### 🎨 モダンなデザイン
- ダークテーマ with Glassmorphism
- スムーズなアニメーション
- レスポンシブレイアウト

## 🚀 セットアップ

### 必要環境
- Node.js 18.x 以上
- npm または yarn

### インストール

```bash
cd c:\src\nn-book
npm install
```

### 開発サーバー起動

```bash
npm run dev
```

ブラウザで `http://localhost:8001/nn-book/` にアクセス

### ビルド

```bash
npm run build
```

## 🎮 使い方

### 1. 入力の調整
- **成績 (Grade)**: 0.0 ~ 1.0
- **態度 (Attitude)**: 0.0 ~ 1.0  
- **応答 (Response)**: 0.0 ~ 1.0
- **目標値**: 不合格 / 保留 / 合格

### 2. 学習の実行
- **1回学習**: Forward → Loss表示 → Backward → Summary
- **自動学習**: 連続学習 (Loss < 0.001で自動停止)
- **手動モード**: 各ステップを手動で進める

### 3. パラメータ調整
- **学習率 (Learning Rate)**: 0.01 ~ 1.0
- **アニメーション速度**: 0.5x ~ 2.0x

## 📁 プロジェクト構造

```
c:\src\nn-book\
├── src/
│   ├── components/          # React コンポーネント
│   │   ├── BackpropModal.tsx      # 逆伝播サマリーモーダル
│   │   └── LossModal.tsx          # Loss計算説明モーダル
│   ├── hooks/
│   │   └── useNeuralNetwork.ts    # NN状態管理フック
│   ├── lib/
│   │   ├── matrix.ts              # 行列演算クラス
│   │   ├── network.ts             # NNメインロジック
│   │   ├── network/
│   │   │   └── backpropagation.ts # 逆伝播ヘルパー関数
│   │   ├── visualizer.ts          # Canvas可視化
│   │   ├── visualizer/
│   │   │   ├── backpropRenderer.ts      # 逆伝播オーバーレイ
│   │   │   ├── calculationOverlay.ts    # 計算ポップアップ
│   │   │   ├── drawingUtils.ts          # 描画ユーティリティ
│   │   │   └── networkRenderer.ts       # ネットワーク描画
│   │   └── types.ts               # TypeScript型定義
│   ├── i18n/
│   │   ├── index.ts               # i18n設定
│   │   └── locales/
│   │       ├── ko.json            # 韓国語翻訳
│   │       └── ja.json            # 日本語翻訳
│   └── App.tsx                    # メインアプリ
├── public/                        # 静的ファイル
└── vite.config.ts                 # Vite設定
```

## 🛠️ 技術スタック

### フロントエンド
- **React 18** - UIフレームワーク
- **TypeScript** - 型安全性
- **Vite** - ビルドツール
- **Material-UI (MUI)** - UIコンポーネント
- **react-i18next** - 国際化

### 可視化
- **Canvas API** - ネットワーク描画
- カスタムレンダラー (Pure TypeScript実装)

### スタイリング
- **Emotion** - CSS-in-JS
- Material-UIテーマカスタマイズ

### 数学ライブラリ
- **自前実装** - PyTorch/TensorFlow不使用
- Pure TypeScriptによるMatrix演算

## 📐 実装の詳細

### ニューラルネットワーク
- **活性化関数**: Sigmoid (隠れ層), Softmax (出力層)
- **損失関数**: Cross-Entropy Loss
- **最適化**: Stochastic Gradient Descent (SGD)
- **学習率**: デフォルト 0.1

### 行列演算
`Matrix`クラスで以下を実装:
- 行列積 (Matrix multiplication)
- 転置 (Transpose)
- 要素ごとの演算 (Element-wise operations)
- スカラー演算

### リファクタリング履歴
- ✅ **Visualizer分離**: 621行 → 311行 (50%削減)
  - `drawingUtils.ts`: 基本描画関数
  - `networkRenderer.ts`: ネットワーク描画
  - `backpropRenderer.ts`: 逆伝播可視化
  - `calculationOverlay.ts`: 計算ポップアップ

- ✅ **Network分離**: 536行 → 260行 (51%削減)
  - `matrix.ts`: 行列演算クラス
  - `backpropagation.ts`: 逆伝播ヘルパー関数

## 🎓 メタファーの対応関係

| ニューラルネット概念 | 面接システムのメタファー |
|---------------------|---------------------|
| 入力ベクトル | 応募者の情報 (成績・態度・応答) |
| ニューロン | 面接官・評価者 |
| 重み | 評価基準・価値観 |
| バイアス | 最低基準・閾値調整 |
| 内積 | 総合評価スコア |
| 活性化関数 | 最終判断の表現 |
| 層 | 面接段階 (1次→2次→最終) |
| 順伝播 | 評価プロセス |
| 誤差逆伝播 | 反省と基準の見直し |
| 学習 | 評価基準の改善 |

## 🤝 Contributing

貢献を歓迎します！Issue・PRをお気軽に。

## 📄 ライセンス

このプロジェクトは教育目的で作成されました。

---

**作成者**: Antigravity AI  
**最終更新**: 2025-12-23
