---
layout : post
title : AIX - Project Outline
---

**이 프로젝트는 크게 다음의 4단계로 구성됩니다.**
<br />
<br />

## **Part 1. 오목 게임 구현**
1. **어떻게 오목을 플레이하나요?**
    - 오목의 규칙에 대해서 소개합니다.
<br />
<br />

2. **강화학습을 위한 state, action**
    - 이후 학습과정에 편리하도록 오목 게임의 상태를 표현합니다.
<br />
<br />

3. **의사코드로 게임을 나타내기**
    - 게임의 진행과정을 의사코드로 간단히 표현합니다.
<br />
<br />

4. **승리를 판정하는 방법**
    - 환경이 돌이 5개 연결된 것을 판정하는 알고리즘을 작성합니다.
<br />
<br />

5. **금지수를 판정하는 방법**
    - 33, 44, 장목과 같은 금지수를 환경이 판정하는 알고리즘을 작성합니다.
<br />
<br />

6. **최적화**
    - Cython을 이용해서 파이썬 코드의 비효율성을 줄입니다.
#
<br />
<br />


## **Part 2. MCTS 구현**
1. **MCTS의 작동원리**
    - mcts가 어떻게 주어진 시간 내에 최선의 전략을 찾아내는지 살펴봅니다.
<br />
<br />

2. **AlphaGo Zero의 MCTS**
    - alphago zero가 어떻게 효율적으로 MCTS를 개선했는지 알아봅니다.
<br />
<br />

3. **MCTS implementation - _Selection_**
    - _Selection_ 단계를 구현합니다.
<br />
<br />

4. **MCTS implementation - _Expansion_ and _Simulation_**
   - _Expansion_ 단계와 _Simulation_단계를 함께 구현합니다.
<br />
<br />

5. **MCTS implementation - _Backpropagation_**
    - _Backpropagation_ 단계를 구현합니다.
#
<br />
<br />

## **Part 3. 신경망**
1. **Convolution layer**
    - 합성곱신경망을 간략하게 알아봅니다.
<br />
<br />

2. **Batch normalization layer**
    - Batch normalization layer을 간략하게 알아봅니다.
<br />
<br />

3. **Residual layer**
    - Residual layer의 구조와 특징을 알아봅니다.
<br />
<br />

4. **Building neural network with PyTorch**
    - pytorch를 사용해서 강화학습에 사용할 신경망을 구현합니다.
#
<br />
<br />

## **Part 4. 강화학습**
1. **정책향상자로서의 MCTS**
    - MCTS가 정책향상자로서 작동할수 있음을 알아봅니다.
<br />
<br />

2. **자가게임**
    - 자가 게임을 진행하여 데이터를 수집하는 과정을 구현합니다.
<br />
<br />

3. **데이터 전처리**
    - 수집된 데이터를 강화학습에 적합하게 전처리 하는 과정을 구현합니다.
<br />
<br />

4. **학습**
    - 학습을 진행합니다.
<br />
<br />

5. **평가**
    - 최종적으로 학습된 신경망을 평가합니다.
#
<br />
<br />

**그럼 시작해봅시다!**
