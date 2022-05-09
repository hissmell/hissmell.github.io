---
layout : post
title : Part 1. 오목 게임 구현
---

## **어떻게 오목을 플레이 하나요?**
![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part01/Omok.png "Example of Omok game")
<br />
<br />
  오목은 흑과 백이 번갈아가면서 돌을 두어가면서 5개가 가로,세로 또는 대각선으로 돌을
 연결하면 승리하는 간단한 규칙을 가지고 있습니다.
<br />
<br />
  하지만 돌을 먼저 두는 흑이 절대적으로 유리하다는 사실 때문에, 흑에게 33, 44, 장목이
 완성되는 자리에는 돌을 둘 수 없다는 제약을 두어 흑백의 균형을 맞춥니다. 이러한 규칙을
 *렌주룰* 이라고 말하며, 대중적으로 가장 널리 사용되는 규칙입니다. 이번 프로젝트에서
 이 렌주룰을 사용하는 오목 게임 환경을 구현합니다.
<br />
<br />
 렌주룰의 규칙은 다음과 같습니다.
- 백은 모든 수를 둘 수 있다.
- 흑은 두어진 수로 *5*가 완성된다면 둘 수 있다.
  * 흑이 두는 수로 인해서 *4*가 2개 이상 완성된다면 그 수는 금지수이다.(44의 경우)
  * 흑이 두는 수로 인해서 *열린 3*이 2개 이상 완성된다면 그 수는 금지수이다.(33의 경우)
  * 흑이 두는 수로 인해서 흑돌이 6개 이상 연결된다면 그 수는 금지수이다.(장목의 경우)

<br />
<br />

* ***4*의 정의**
<br />
<br />

 *4*는 돌을 한번 더 두어서 5개를 연결할 수 있는 경우를 의미합니다.
<br />
<br />
![Example of 4]({{ site.baseurl }}/images/AIX_Project_Part01/4_01.PNG)
<br />
<br />
* ***열린 3*의 정의**
<br />
<br />

  *열린 3*은 돌을 한번 더 두어서 *열린 4*를 만들 수 있는 상태를 의미합니다.
<br />
<br />
![Example of 33]({{ site.baseurl }}/images/AIX_Project_Part01/33_01.PNG)
<br />
<br />
  위 사진은 대표적인 33 금지수의 예시입니다. 빨간색 X 표시 위치에 흑이 돌을 두게 된다
 면 흑은 *열린 3* 2개가 동시에 완성되게 됩니다. 따라서 금지수를 두게 된 흑은 패배처리
 됩니다.
<br />
<br />
![Example of 33]({{ site.baseurl }}/images/AIX_Project_Part01/33_02.PNG)
<br />
<br />
  위 사진은 오인하기 쉬운 33 금지수 위치의 예시입니다. 언뜻 보기에는 빨간색 X 표시 위
 치에 돌을 두면 *열린 3*이 2개가 동시에 생겨서 금지수인것 같지만, 아래쪽 3이라고 생각
 되는 곳은 돌을 하나 더 두었을 때 *열린 4*를 만들 수 없는 위치입니다. 따라서 금지수가
 아닌 둘 수 있는 곳입니다.
<br />
<br />
* ***장목*의 정의**
<br />
<br />

  장목은 같은 색의 돌이 6개 이상 연결되는 경우를 의미합니다.
<br />
<br />
![Example of 33]({{ site.baseurl }}/images/AIX_Project_Part01/long_01.PNG)
<br />
<br />

## **강화학습을 위한 State와 Action**
   강화학습을 효과적으로 진행하기 위해서, 학습에 적합한 State와 Action 표현을 설정하는
 것은 반드시 선행되어야 하는 과제입니다. State와 Action은 다음의 요소를 포함하는 것이
  바람직합니다.
<br />
<br />
- **State representation**
    * **관측 가능한 Environment의 정보**를 모두 표현한다.
    * 가능하다면 **Markov decision process(MDP)**가 되도록 State를 표현한다.
  간단히 말하면, **환경의 다음 상태가 오로지 이전의 상태 정보와 입력된 action에만 의존**
  하도록 State를 표현한다.
    * 사용되는 메모리를 최소화한다.

- **Action representation**
    * 가능한 한 적은 메모리와 연산에 적합한 표현을 가져야 한다.
<br />
<br />

 위의 조건에 맞춰서 이번 프로젝트에서는 state와 action을 다음과 같이 표현하였습니다.

- **State**
    * Type : numpy.array, np.float32
    * Shape : (board_size,board_size,2); pytorch를 사용하기 때문에 채널이
  마지막 차원에 위치합니다.
    * Description : 첫번째 채널에는 해당 State에서 누구의 턴인지를 표시합니다. 만약 흑
  의 차례라면 -1로 차있고, 백의 차례라면 1로 가득 차 있습니다.  
  두번째 채널은 보드에서 돌의 위치를 표시합니다. 흑돌이 놓여진 곳은 -1, 백돌이 놓여진 곳은
  1, 돌이 놓여져 있지 않은 곳은 0으로 표기되어 있습니다.

- **Action**
    * Type : numpy.int64
    * Range : range(0,board_size * board_size)
    * Description : Action을 board_size로 나눈 몫을 돌을 두는 곳의 행, 나머지를
  열로 사용합니다.

## 의사코드로 게임을 나타내기
 **게임의 진행과정을 의사코드로 나타내면 다음과 같습니다**
1. 흑 또는 백이 action을 합니다.
<br />

2. action에 해당하는 위치에 돌이 이미 존재하거나 board의 범위를 벗어나는 위치라면 에러를 반환합니다.
<br />

3. action으로 인해서 *5*가 완성된다면 결과값을 반환합니다.
<br />

4. 백 차례라면, action에 해당하는 위치에 돌을 두고 다음 State를 반환합니다.
<br />

5. action으로 인해서 *장목*이 완성된다면 흑에게 패배처리를 하고 종료합니다.
<br />

6. action으로 인해서 *44*가 완성된다면 흑에게 패배처리를 하고 종료합니다.
<br />

7. action으로 인해서 *33*이 완성된다면 흑에게 패배처리를 하고 종료합니다.
<br />

8. action에 해당하는 위치에 돌을 두고 다음 State를 반환합니다.
<br />

9. 보드에 전부 돌이 놓여지거나 *5*가 완성되면 그에 해당하는 결과값을 반환하고 게임을 종료합니다.
<br />

## **승리를 판정하는 방법**

<br />
<br />

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part01/win_01.PNG "Black win!")

 * **승리조건** : 같은 색의 돌을 가로 또는 세로 또는 대각 방향으로 5개 연결하면 승리

<br />

같은 색의 돌이 5개 연결되어 있는지 판단하는 알고리즘은 다음과 같습니다. 간략한 설명을
위해서 가로방향으로 5개 연결되어 있는 것을 확인합니다. 세로, 대각의 경우도 같은 방식으
로 확인 할 수 있습니다.
<br />
<br />
1. 돌을 놓는 위치로 *탐색*위치를 이동합니다.
2. *탐색*위치에서 왼쪽으로 *탐색*위치를 이동하고 그 위치를 *탐색*합니다.
3. *탐색* 결과, 해당 플레이어의 돌이 놓여져 있다면 find_stones 변수에 1을 더하고 왼쪽으로 한칸 이동해서 다시 *탐색*을 반복합니다.
4. *탐색* 결과, 상대 플레이어의 돌이 놓여져 있거나, 비어있는 자리라면 처음 돌을 놓았던 위치로 *탐색*위치로 이동합니다.
5. 왼쪽으로 *탐색*을 진행하다가 상대의 돌 또는 빈 자리를 *탐색*하여 처음자리로 돌아오면 오른쪽으로 위의 *탐색*과정을 반복합니다.
6. 위 (1) ~ (5)과정이 끝나고 find_stones 가 5가 되었다면 해당 플레이어가 승리하였다고 판정합니다.
<br />

위의 알고리즘을 세로 방향과 대각선 방향들로도 진행하면 주어진 action이 *5*를 만들었는지를
 판단할 수 있습니다.

 * 코드 : 위 알고리즘을 파이썬 코드로 옮깁니다.

```python
check_color = turn
opponent_color = -turn
max_streak = 0
count_4 = 0
# 가로 체크
y = check_y + 1
streak = 1 # 탐색 시작 위치에는 항상 플레이어의 돌이 놓여져 있기에 1에서 시작합니다.
while 1:
    if y >= board_size or y < 0:
        break
    if board_array[check_x][y] == opponent_color or board_array[check_x][y] == 0:
        break
    if board_array[check_x][y] == turn:
        streak += 1
    y += 1

y = check_y - 1
while 1:
    if y >= board_size or y < 0:
        break
    if board_array[check_x][y] == opponent_color or board_array[check_x][y] == 0:
        break
    if board_array[check_x][y] == turn:
        streak += 1
    y -= 1

if max_streak < streak:
    max_streak = streak

if max_streak == 5:
    return 2 # 5가 완성되었음을 의미합니다.
```

## **금지수를 판정하는 방법**


