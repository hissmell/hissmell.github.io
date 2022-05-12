---

layout : post
title : Part 2. MCTS 구현

---

**이번 포스트에서는 MCTS를 구현하는 과정을 소개합니다**

---

## **MCTS의 작동 원리**

 * MCTS *Monte Carlo Tree Search*의 탐색과정

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part02/mcts01.png "MCTS progress scheme")
<br />
<br />
 MCTS의 탐색과정은 다음의 4가지 단계로 이루어집니다.
<br />
<br />
 1. ***Selection***
<br />

 기존에 가지고 있는 탐색트리 내에서 트리 정책을 통해서 리프노드를 선택합니다.
 트리 정책은 알고리즘마다 세세한 부분은 차이가 있을 수 있지만, 대개의 경우, 트리정책은 해당 노드의
 플레이어가 자신의 이득을 최대로 할 수 있는 노드를 선택하는 것을 의미하도록 설정됩니다.


 2. ***Expansion***
<br />

 *Selection* 단계에서 도달한 리프노드에서 자식노드를 추가합니다. 추가된 노드의 추정 가치나
 방문횟수같은 통계 값들은 모두 0으로 설정됩니다.


 3. ***Simulation***
<br />

 *Expansion* 단계에서 추가된 자식노드의 추정 가치를 결정하는 단계입니다. 자식노드에서
 시작하여, 완전 무작위행동을 하는 정책, 또는 roll out policy 라는 특정 정책을 사용하여
 그 노드에서 게임을 끝까지 진행하였을 때 어떤 결과가 나오는지 확인합니다. 시뮬레이션을 한
 결과를 그 노드의 가치 추정값으로 사용합니다.  
 시뮬레이션을 하는 횟수가 증가할 수록 노드의 가치 추정값은 정확해지며, roll out policy가
 우수한 성능을 가질 수록 그 노드의 가치 추정값이 정확해집니다.


 4. ***Backpropagation***
<br />

 추가된 노드의 가치 추정값, 또는 방문횟수와 같은 모든 통계적 데이터들을 그 노드에 도달하기
 위해서 거쳤던 모든 노드에 합산합니다.

<br />
<br />

 (1) ~ (4)의 모든 과정을 여러번 반복하면서 트리는 점점 확장되고, 트리가 가지고 있는
 노드의 가치 추정값들은 점차 정확해집니다. 따라서 MCTS를 통해서 행동하는 AI는 단순히 rollout
 policy를 사용하는 AI에 비해서 더 성능이 우수합니다.

<br />
<br />

---

## **Alphago Zero의 MCTS**
<br />
<br />

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part02/mcts01.png "MCTS progress scheme")
<br />
<br />
알파고 제로는 인간의 지식을 사용하지 않고, 오로지 MCTS와 자가대국, 강화학습만으로 인간의
실력을 아득히 뛰어넘은 바둑 AI입니다.  
알파고 제로의 등장 이전에도 바둑을 MCTS를 사용해서 해결하려는 AI가 없었던 것은 아니었지만
바둑의 특성상 너무나 많은 경우의 수를 효율적으로 탐색하기에는 무리가 있었습니다.  
알파고 제로는 기존의 MCTS를 **탐색 Depth**와 **탐색 Width** 2가지 측면으로 개선하여
바둑을 해결할 수 있었습니다.
<br />
<br />

* **탐색 Depth 개선**
<br />

기존의 MCTS는 새로 추가되는 노드의 가치를 추정하기 위해서 그 노드를 시작으로 *Simulation*
을 여러번하여 게임을 끝까지 할 때 나오는 결과를 평균내어야 했습니다. 하지만 바둑은
게임의 결과를 얻기까지 (게임의 초반이라면) 약 300수 정도를 더 두어야 하고, 이렇게 결과상태까지의  
 거리가 멀다면 얻게된 결과도 노드의 가치와 연관성이 매우 약하게 됩니다. 때문에 정확한 가치 추정값을
 얻기위해서는 매우 많은 *Simulation*이 필요하고, 이는 짧은 시간내에 문제를 해결할 수 없는
 원인이 됩니다.  
 알파고 제로는 *Simulation*단계를 신경망에 해당 노드의 State를 입력하여 가치를 추정하는
 것으로 대체합니다. 이렇게 함으로써 탐색의 깊이를 게임의 끝까지 하지 않고 효율적으로
 줄일 수 있습니다.

 * **탐색 Width 개선**
<br />

한 상태에서 가능한 action의 가짓수(branch factor)가 크다는 것도 MCTS에서 문제가
됩니다. 바둑의 가능한 action의 수를 대략 361이라고 한다면 N 단계까지 탐색을 진행하기
위해서는 361^N 개의 노드가 완성되어야 합니다. 알파고 제로는 이러한 문제를 트리정책을
학습시켜서 해결하였습니다. 한 상태에서 전이가능한 다음 상태들에 대해서, *그럴듯한* 상태에
 높은 확률을 부여해서 *Selection*단계에서 방문할 확률이 높도록 설정하였습니다. 이러한
 효과로, 트리가 모든 상태를 균등하게 탐색하지 않고, 유력한 상태를 주로 탐색하면서 탐색의
 효율을 증가시켰습니다.

<br />
<br />

---

## **MCTS 구현 - *Selection***

<br />
<br />

 트리의 노드는 딕셔너리에 의해서 관리됩니다.

```python
def find_leaf(self, state_np, player):
    '''
    :param state_np:
    :param player:
    (1) value : None if leaf node, or else equals to the game outcome for the player at leaf node
    (2) leaf_state_np : nd_array bytes of the leaf node
    (3) player : player at the leaf node
    (4) states_b : list of states (in bytes) traversed
    (5) actions : list of actions (int 64) taken
    :return:
    '''
    states_b = []
    actions = []
    cur_state_np = state_np
    cur_player = player
    value = None
    env_copy = copy.deepcopy(self.env)

    while not self.is_leaf(cur_state_np):
        cur_state_b = cur_state_np.tobytes()
        states_b.append(cur_state_b)
        counts = self.visit_count[cur_state_b]
        total_sqrt = math.sqrt(sum(counts))
        probs = self.probs[cur_state_b]
        values_avg = self.values_avg[cur_state_b]

        # if cur_state_b == state_b:
        #     noises = np.random.dirichlet([0.03] * self.action_size)
        #     probs = [0.75 * prob + 0.25 * noise for prob,noise in zip(probs,noises)]

        score = [q_value + self.c_punc * prob * total_sqrt / (1 + count)
                 for q_value,prob,count in zip(values_avg,probs,counts)]

        illegal_actions = set(self.all_action_list) - set(env_copy.legal_actions)
        for illegal_action in illegal_actions:
            score[illegal_action] = -np.inf

        for waiting_action in self.waiting[cur_state_b]:
            score[waiting_action] = -100

        action = int(np.argmax(score))
        actions.append(action)

        cur_state_np, reward, done, info = env_copy.step(action)
        cur_player = -cur_player
        if done:
            if cur_player == BLACK:
                value = reward
            else:
                value = -reward

    return value, cur_state_np, cur_player, states_b, actions
```

<br />
<br />

---

## **MCTS 구현 - *Expansion* and *Simulation***

<br />
<br />

 *Simulation*은 신경망이 가치를 추정하는 단계로 대체됩니다.

```python
backup_queue = []
expand_states_np = []
expand_players = []
expand_queue = []

for i in range(count):
    value, leaf_state_np, leaf_player, states_b, actions = self.find_leaf(state_np,player)
    leaf_state_b = leaf_state_np.tobytes()
    if value is not None:
        backup_queue.append((value,states_b,actions))
    else:
        self.waiting[states_b[-1]].append(actions[-1])
        expand_states_np.append(leaf_state_np)
        expand_players.append(leaf_player)
        expand_queue.append((leaf_state_b,states_b,actions))

if expand_queue:
    if not net == None:
        batch_var = torch.tensor(np.array(expand_states_np), dtype=torch.float32).to(device)
        logits_var, values_var = net(batch_var)
        probs_var = F.softmax(logits_var,dim=1)
        values_np = values_var.data.to('cpu').numpy()[:,0] # 흑 플레이어 입장에서 value입니다.
        probs_np = probs_var.data.to('cpu').numpy()
    else:
        N = np.array(expand_states_np).shape[0]
        probs_np = np.ones([N,self.action_size],dtype=np.float32) / self.action_size
        values_np = np.ones([N],dtype=np.float32) / self.action_size

    # 흑 플레이어 입장에서 예측된 value를 해당 노드 플레이어 입장에서 value로 조정합니다.
    for ii in range(len(expand_players)):
        values_np[ii] = values_np[ii] if expand_players[ii] == BLACK else -values_np[ii]

    for (leaf_state_b,states_b,actions),value,prob in zip(expand_queue,values_np,probs_np):
        self.visit_count[leaf_state_b] = [0] * self.action_size
        self.value[leaf_state_b] = [0.0] * self.action_size
        self.values_avg[leaf_state_b] = np.random.uniform(low=-0.01,high=0.01,size=self.action_size)
        self.probs[leaf_state_b] = prob
        backup_queue.append((value,states_b,actions))
self.waiting.clear()
```

<br />
<br />

---

## **MCTS 구현 - *Backpropagation***

<br />
<br />

한가지 주의해야 할 점은, 이 구현과정에서 게임의 최종상태, 즉 *5*가 완성되거나 완성되지
못한 채로 돌로 보드가 가득차있는 상태는 노드에 포함되지 않습니다.  
오목은 2명이 적대적으로 진행되는 게임이므로 어떤 상태의 가치가 1이라면 그 상태는 이전
플레이어 입장에서는 가치가 -1것과 같이 간주됩니다.
<br />
<br />

```python
for value,states_b,actions in backup_queue:
    '''
    leaf state is not stored in states_b.
    therefore value is supposed to opponent's value.
    so we have to convert it to reverse!
    '''
    cur_value = -value
    for state_b, action in zip(states_b[::-1],actions[::-1]):
        self.visit_count[state_b][action] += 1
        self.value[state_b][action] += cur_value
        self.values_avg[state_b][action] = self.value[state_b][action] / self.visit_count[state_b][action]
        cur_value = -cur_value
```

<br />
<br />

---

## **최적화**

위 4개의 단계 중에서 가장 시간이 가장 많이 소모되는 과정은 *Simulation* 단계입니다.
**인공신경망을 호출하는 것은 상당히 비싼연산이기 때문에 매번 *Expansion*을 할 때마다
인공신경망을 호출하는 것은 MCTS의 탐색속도를 극히 느리게 만드는 주요한 요인이 됩니다.**  
이러한 문제를 해결하기 위해서 약간의 정확성을 희생하지만, 미니배치를 만들어 여러 노드를
한번에 트리에 추가하는 것이 도움이 될 수 있습니다.  
 이 프로젝트에서는 미니배치를 만들어서 MCTS를 구현하였으며, 8개의 노드를 한번에 확장하는
 방식으로 진행하였습니다.
 <br />
 <br />

 ---
 ***

 MCTS의 완성된 파일은 [이곳](https://github.com/hissmell/Pytorch_Toy_Projects/blob/Omok/Omok/lib/mcts.py)
 에서 확인 할 수 있습니다.

<br />
<br />

---
***

