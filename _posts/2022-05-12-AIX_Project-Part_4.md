---
layout : post
title : Part 4. 강화학습
--------------------

이 포스트에서는 MCTS와 자가게임이 강화학습에 적용되는 과정을 구현합니다.
<br />
<br />

---

## **정책향상자로서의 MCTS**
<br />

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part04/mcts01.PNG)
 충분한 탐색을 동반한 MCTS으로 산출되는 정책은 기존의 정책에 비해서 대부분의 경우, 강한
 정책입니다. 따라서 다음의 과정으로 학습을 진행하면 향상된 정책을 얻을 수 있습니다.
<br />
<br />
![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part04/mcts02.png)
<br />
<br />

1. 기존의 정책을 MCTS의 트리정책으로 사용합니다.
<br />

2. 자가 게임을 통해서 게임 데이터를 수집합니다.
<br />

3. 수집된 게임 데이터를 학습 데이터로 활용하여, 기존의 정책이 수집된 게임 데이터 속 정책처럼
행동하도록 지도학습을 진행합니다.
<br />

4. 학습된 정책을 다시 MCTS의 트리정책으로 사용하여 위 과정을 반복합니다.
<br />

---

## **자가 게임**
<br />

자가 게임을 하는 함수 play_game을 구현합니다!  
입력 변수들의 의미는 다음과 같습니다.
<br />
<br />
* **env** : 게임을 하는 환경입니다.
<br />

* **mcts_stores** : 게임에 사용할 mcts 트리를 입력합니다.
<br />

* **replay_buffer** : 게임 데이터를 저장할 queue를 입력합니다.
<br />

* **net1,net2** : 각 트리가 사용하는 신경망 1,2 입니다.
<br />

* **steps_before_tau_0** : 탐색적으로 AI가 행동하는 마지막 스텝입니다.
<br />

* **mcts_searches** : expand하는 횟수입니다.
<br />

* **mcts_batch_size** : 한번 expand할 때 노드를 확장하는 개수입니다.
<br />

* **net1_plays_first** : 이 값이 참이면 첫번째 신경망이 항상 흑으로 시작합니다.
<br />

* **device** : 신경망이 cpu에서 동작할지, gpu(cuda)에서 동작할지를 결정합니다.
<br />

* **render** : 이 값이 참이라면 게임 과정을 출력합니다.
<br />

* **return_history** : 이 값이 참이면 게임 데이터를 함수가 반환합니다.
<br />

* **gamma** : 미래의 reward의 discount fator입니다.
<br />

코드는 아래와 같습니다.
<br />
<br />
```python

def play_game(env,mcts_stores,replay_buffer,net1,net2
              ,steps_before_tau_0,mcts_searches,mcts_batch_size
              ,net1_plays_first=False,device='cpu',render=False
              ,return_history=False,gamma=1.0):
    """
    Play one single game, memorizing transitions into the replay buffer
    """

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(env),mcts.MCTS(env)]
    elif isinstance(mcts_stores,mcts.MCTS):
        mcts_stores = [mcts_stores,mcts_stores]

    state = env.reset()
    nets = [net1.to(device),net2.to(device)]

    if not net1_plays_first:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0
    if cur_player == 0:
        net1_color = -1
    else:
        net1_color = 1

    step = 0
    tau = 0.08 if steps_before_tau_0 <= step else 1
    game_history = []

    result = None
    net1_result = None
    turn = -1 # (-1) represents Black turn! (1 does White turn)
    while result is None:
        if render:
            print(env.render())

        mcts_stores[cur_player].update_root_node(env,state,nets[cur_player],device=device)
        mcts_stores[cur_player].search_batch(mcts_searches,mcts_batch_size
                                             ,state,turn,nets[cur_player]
                                             ,device=device)
        probs,_ = mcts_stores[cur_player].get_policy_value(state,tau=tau)
        game_history.append((state,cur_player,probs))
        action = int(np.random.choice(mcts_stores[0].all_action_list, p=probs))

        if render:
            if turn == -1:
                if net1_color == -1:
                    print(colored(f"Turn : Net 1 (O turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 2 (O turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            else:
                if net1_color == -1:
                    print(colored(f"Turn : Net 2 (X turn) Nodes {len(mcts_stores[cur_player].probs):}",'blue'))
                else:
                    print(colored(f"Turn : Net 1 (X turn) Nodes {len(mcts_stores[cur_player].probs):}", 'blue'))
            N_dict, Q_dict = mcts_stores[cur_player].get_root_child_statistics()
            top = min(3, len(env.legal_actions))
            N_list = sorted(list(N_dict.keys()), key=lambda x: N_dict[x], reverse=True)
            for i in range(1, top + 1):
                print(colored(
                    f'Top {i} Action : ({N_list[i - 1][0]:d},{N_list[i - 1][1]:d})'
                    f' Visit : {N_dict[N_list[i - 1]]} Q_value : {Q_dict[N_list[i - 1]]:.3f}'
                    f' Prob : {probs[env.encode_action(N_list[i - 1])]*100:.2f} %','cyan'))
            move = env.decode_action(action)
            print(colored(f"Action taken : ({move[0]:d},{move[1]:d})"
                          f" Visit : {N_dict[move]} Q_value : {Q_dict[move]:.3f}"
                          f" Prob : {probs[env.encode_action(move)]*100:.2f} %",'red'))


        state,reward,done,_ = env.step(action)
        if done:
            if render:
                print(env.render())
            result = reward
            if net1_color == -1:
                net1_result = reward
            else:
                net1_result = -reward

        cur_player = 1 - cur_player
        turn = -turn

        step += 1
        if step >= steps_before_tau_0:
            tau = 0.08


    h = []
    if replay_buffer is not None or return_history:
        for state, cur_player, probs in reversed(game_history):
            if replay_buffer is not None:
                replay_buffer.append((state, cur_player, probs, result))
            if return_history:
                h.append((copy.deepcopy(state), cur_player, probs, result))

            result = -result * gamma

    return net1_result, step, h
```
<br />
<br />

---

## **데이터 전처리**
