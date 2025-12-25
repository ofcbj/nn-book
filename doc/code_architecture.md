# Neural Network Visualization - Code Architecture

## 전체 구조 다이어그램

```mermaid
flowchart TB
    subgraph Entry["Entry Point"]
        main["main.tsx"]
        App["App.tsx"]
    end

    subgraph Components["React Components"]
        Header["Header.tsx"]
        ControlPanel["ControlPanel.tsx"]
        NetworkCanvas["NetworkCanvas.tsx"]
        StatsDisplay["StatsDisplay.tsx"]
        CalcPanel["CalculationPanel.tsx"]
        Heatmap["ActivationHeatmap.tsx"]
        
        subgraph Modals["Modal Components"]
            LossModal["LossModal.tsx"]
            BackpropModal["BackpropModal.tsx"]
            HelpModal["HelpModal.tsx"]
            WeightModal["WeightComparisonModal.tsx"]
        end
    end

    subgraph Hooks["React Hooks"]
        useNN["useNeuralNetwork.ts"]
        useAnim["useAnimationStateMachine.ts"]
    end

    subgraph CoreLib["Core Library (lib/)"]
        Network["network.ts<br/>NeuralNetwork Class"]
        Matrix["matrix.ts<br/>Matrix Operations"]
        Types["types.ts<br/>Type Definitions"]
        AnimState["animationState.ts<br/>State Machine"]
        AnimLoop["animationLoop.ts<br/>Animation Utils"]
        Colors["activationColors.ts"]
        WeightComp["weightComparison.ts"]
        
        subgraph NetworkSub["network/"]
            Backprop["backpropagation.ts"]
        end
    end

    subgraph Visualizer["Visualizer Module (lib/visualizer/)"]
        VisMain["visualizer.ts<br/>Main Visualizer Class"]
        NetRenderer["networkRenderer.ts"]
        DrawUtils["drawingUtils.ts"]
        BackRenderer["backpropRenderer.ts"]
        CalcOverlay["calculationOverlay.ts"]
        OverlayContent["overlayContentGenerator.ts"]
        OverlayRenderer["overlayRenderer.ts"]
    end

    subgraph I18n["Internationalization"]
        i18nIndex["i18n/index.ts"]
        Locales["locales/<br/>ko.json, en.json, ja.json"]
    end

    %% Connections
    main --> App
    App --> Components
    App --> useNN
    
    useNN --> useAnim
    useNN --> Network
    useNN --> VisMain
    
    Network --> Matrix
    Network --> Backprop
    Network --> Types
    
    useAnim --> AnimState
    useNN --> AnimLoop
    
    VisMain --> NetRenderer
    VisMain --> BackRenderer
    VisMain --> CalcOverlay
    
    NetRenderer --> DrawUtils
    BackRenderer --> OverlayContent
    BackRenderer --> OverlayRenderer
    CalcOverlay --> OverlayContent
    CalcOverlay --> OverlayRenderer
    
    Components --> i18nIndex
    OverlayContent --> i18nIndex
```

---

## 모듈별 책임

### 🎯 Entry & App
| 파일 | 책임 |
|------|------|
| `main.tsx` | React 앱 진입점, 테마 적용 |
| `App.tsx` | 메인 레이아웃, 컴포넌트 조합 |

---

### 🧩 Components
| 컴포넌트 | 책임 |
|----------|------|
| `Header` | 제목, 도움말 버튼 |
| `ControlPanel` | 입력 슬라이더, 학습 버튼 |
| `NetworkCanvas` | 신경망 시각화 캔버스 |
| `StatsDisplay` | 에포크, 손실, 예측 표시 |
| `CalculationPanel` | 연산 과정 텍스트 표시 |
| `ActivationHeatmap` | 레이어별 활성화 히트맵 |
| `LossModal` | Forward Pass 결과, Loss 계산 표시 |
| `BackpropModal` | 역전파 완료 요약 |
| `HelpModal` | 사용법 안내 |
| `WeightComparisonModal` | 학습 전후 가중치 비교 |

---

### 🪝 Hooks
| Hook | 책임 |
|------|------|
| `useNeuralNetwork` | 신경망 생성, 학습, 애니메이션 로직 총괄 |
| `useAnimationStateMachine` | Forward/Backward 애니메이션 상태 관리 |

---

### 🧠 Core Library

#### Network
| 파일 | 책임 |
|------|------|
| `network.ts` | NeuralNetwork 클래스 (feedforward, train, getCalculationSteps) |
| `network/backpropagation.ts` | 역전파 계산 로직 (createBackpropSteps) |
| `matrix.ts` | 행렬 연산 (Matrix 클래스) |
| `types.ts` | 모든 타입 정의 |

#### Animation
| 파일 | 책임 |
|------|------|
| `animationState.ts` | 애니메이션 상태 머신 (idle, forward_animating, backward_animating, etc.) |
| `animationLoop.ts` | 공통 애니메이션 루프 유틸리티 |

#### Utils
| 파일 | 책임 |
|------|------|
| `activationColors.ts` | 활성화값 → 색상 변환 |
| `weightComparison.ts` | 가중치 변화량 계산 |

---

### 🎨 Visualizer Module

```mermaid
flowchart LR
    subgraph Main["메인"]
        Vis["visualizer.ts<br/>Visualizer Class"]
    end
    
    subgraph Rendering["렌더링"]
        Net["networkRenderer.ts<br/>전체 네트워크 렌더링"]
        Draw["drawingUtils.ts<br/>뉴런/연결선 그리기"]
        Back["backpropRenderer.ts<br/>역전파 하이라이트"]
    end
    
    subgraph Overlay["오버레이"]
        Calc["calculationOverlay.ts<br/>Forward 팝업"]
        Content["overlayContentGenerator.ts<br/>팝업 내용 생성"]
        Render["overlayRenderer.ts<br/>팝업 박스 렌더링"]
    end
    
    Vis --> Net
    Vis --> Back
    Vis --> Calc
    Net --> Draw
    Back --> Content
    Back --> Render
    Calc --> Content
    Calc --> Render
```

---

### 🌐 Internationalization
| 파일 | 책임 |
|------|------|
| `i18n/index.ts` | i18next 설정 |
| `locales/ko.json` | 한국어 번역 |
| `locales/en.json` | 영어 번역 |
| `locales/ja.json` | 일본어 번역 |

---

## 데이터 흐름

```mermaid
sequenceDiagram
    participant User
    participant ControlPanel
    participant useNeuralNetwork
    participant NeuralNetwork
    participant Visualizer
    participant Canvas

    User->>ControlPanel: 입력값 조정 / 버튼 클릭
    ControlPanel->>useNeuralNetwork: 상태 업데이트
    useNeuralNetwork->>NeuralNetwork: feedforward() / train()
    NeuralNetwork-->>useNeuralNetwork: 계산 결과
    useNeuralNetwork->>Visualizer: update(nn)
    Visualizer->>Canvas: drawNetwork()
    Canvas-->>User: 시각화 표시
```

---

## 애니메이션 상태 흐름

```mermaid
stateDiagram-v2
    [*] --> idle
    
    idle --> forward_animating: startAnimation()
    forward_animating --> forward_animating: forwardTick()
    forward_animating --> show_loss_modal: forwardComplete()
    
    show_loss_modal --> backward_animating: startBackward()
    backward_animating --> backward_animating: backwardTick()
    backward_animating --> show_backprop_complete: backwardComplete()
    
    show_backprop_complete --> idle: close modal
    
    forward_animating --> idle: stop()
    backward_animating --> idle: stop()
```
