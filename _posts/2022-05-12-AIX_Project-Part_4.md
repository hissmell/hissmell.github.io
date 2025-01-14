---
layout : post
title : Part 4. 강화학습
---

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
<br />

 데이터 전처리를 하는 목적은 **단순히 강화학습을 진행했을 때 발생할 수 있는 여러가지  
 문제**들을 방지하기 위함입니다.
<br />
<br />
 전처리 없이 강화학습을 진행하면 크게 2가지의 문제점이 발생하게 됩니다.

* **학습 데이터의 편향된 분포**
 완전히 학습을 처음 시작할때(iter_num=0), 신경망은 입력된 데이터에 대해서 무작위 정책,
무작위 가치를 산출하게 됩니다. 한번 움직일때마다 AI가 MCTS로 탐색하는 횟수가 약 800회
이기 때문에, 이 AI는 앞으로 1수정도만을 유효하게 움직입니다. 따라서 종료된 게임에서 두어진
대부분의 움직임들은 무작위적으로 움직인 결과이고, 게임의 최종결과와는 거의 무관한 데이터
가 됩니다.  
 한 게임의 평균 길이를 30정도라고 한다면, 유효한 데이터는 1개 정도이고 나머지는 무작위
움직임의 결과로 나온 데이터이기 때문에 이 데이터를 그대로 학습에 활용한다면 무작위 데이터를
산출하는 방향으로 학습될 가능성이 매우 높습니다.
<br />

* **데이터 간 높은 correlation에 의한 오버피팅**
 오목의 게임데이터는 한스텝이 진행될때마다 돌의 위치가 하나씩만 변하므로 데이터간의 상관
관계가 매우 높습니다. 이러한 높은 상관관계로 인해서 게임의 데이터를 모두 학습에 활용하게
된다면 AI가 게임의 일반적인 특성을 추출하지 않고, 게임의 데이터가 어떤 게임이었는지,
그 게임의 결과가 무엇이었는 학습하는 방향으로 학습하게 됩니다. 즉, 오버피팅이 발생하게 됩니다.
<br />

 위의 2문제들은 학습에 치명적인 장애를 초래합니다. 따라서 다음과 같은 전처리를 통해서
 위의 2문제들을 방지하였습니다.
<br />
<br />
* **한 게임에서 소수의 샘플 데이터만 추출**
 한 게임의 모든 데이터를 사용하게 되면 데이터간의 높은 상관관계로 오버피팅이 발생합니다.
한 게임에서 소수의 샘플 데이터를 랜덤으로 추출해서 학습에 사용한다면 이 문제를 해결할 수
있습니다. 한 게임당 1개의 샘플 데이터만을 추출하는 것이 이상적이겠지만, 그렇게 한다면
학습에 필요한 게임수가 너무 많아지기 때문에 이번 프로젝트에서는 한 게임당 2개의 샘플을
랜덤으로 추출하였습니다.
<br />

* **첫번째 학습때에, 마지막 스텝은 확정 추출**
 첫번째 학습에서 학습의 안정성을 위해서, 추출하는 2개의 샘플을 하나는 게임의 마지막 스텝,
나머지 하나는 게임의 중간에서 추출합니다. 이렇게 하면 학습의 속도를 높일 수 있습니다.
<br />

* **Discount factor의 조절**
 Discount factor(gamma)를 1 - (1/iter_num)^2으로 설정합니다. iteration을 반복
할 수록 AI의 움직임이 게임의 결과에 미치는 영향이 증가하기 때문에 감마를 이렇게 설정하여
학습의 안정성을 높입니다.
<br />

---

## **최적화**
<br />

 자가게임을 통해 데이터를 수집하는 과정은 학습과정에서 가장 많은 시간을 소비하게 되는
 과정입니다. 이 과정을 파이썬의 multiprocessing 라이브러리를 통해서 병렬화하면
 병렬화된 프로세스의 개수만큼 데이터를 더 빠르게 수집할 수 있습니다.
 <br />
 <br />

 ```python
import multiprocessing as mp

 # 멀티프로세싱으로 데이터 수집 가속
def mp_collect_experience(max_game_num,path_dict,env,local_net,name,gamma,device):
    while True:
        if len(os.listdir(path_dict['data_dir_path'])) >= max_game_num:
            break
        mcts_stores = mcts.MCTS(env)
        t = time.time()

        _, game_steps, game_history = common.play_game(env, mcts_stores, replay_buffer=None
                                                  ,net1=local_net, net2=local_net
                                                  ,steps_before_tau_0=STEPS_BEFORE_TAU_0
                                                  ,mcts_searches=MCTS_SEARCHES
                                                  ,mcts_batch_size=MCTS_BATCH_SIZE, device=device
                                                  ,render=False,return_history=True,gamma=gamma)
        dt = time.time() - t
        step_speed = game_steps / dt
        node_speed = len(mcts_stores) / dt
        print(colored(f"------------------------------------------------------------\n"
                      f"(Worker : {name})\n"
                      f" Game steps : {len(os.listdir(path_dict['data_dir_path']))}"
                      f" Game length : {game_steps}\n"
                      f"------------------------------------------------------------", 'red'))
        print(colored(f"  * Used nodes in one game : {len(mcts_stores) // PLAY_EPISODE:d} \n"
                      f"  * Game speed : {step_speed:.2f} moves/sec ||"
                      f"  Calculate speed : {node_speed:.2f} node expansions/sec \n"
                      , 'cyan'))

        game_path = os.path.join(path_dict['data_dir_path'], f"game_{len(os.listdir(path_dict['data_dir_path'])):d}")
        state_list = []
        probs_list = []
        result_list = []
        for state_arr,_,probs_arr,result in reversed(game_history):
            state_list.append(state_arr)
            probs_list.append(probs_arr)
            result_list.append(result)
        state_list = np.stack(state_list,axis=0).tolist()
        probs_list = np.stack(probs_list,axis=0).tolist()
        game_data = {'states':state_list,'probs':probs_list,'results':result_list}
        with open(game_path+'.json','w') as f:
            json.dump(game_data,f)
        del mcts_stores

while True:
    for _ in range(PLAY_EPISODE):
        # 멀티 프로세스들마다 게임 데이터 수집
        workers = [mp.Process(target=mp_collect_experience,
                              args=(MAX_GAME_NUM,path_dict,env,net,f"Worker{i:02d}",
                                    GAMMA,
                                    device)) for i in range(NUM_WORKERS)]

        for i in range(NUM_WORKERS):
            workers[i].start()
            time.sleep(100)
        [worker.join() for worker in workers]
        [worker.close() for worker in workers]
 ```
 <br />
 <br />

 제 노트북에서는 램 용량이 부족해서 병렬화된 프로세스를 3개로 설정하였지만 램과 그래픽카드의
 메모리가 충분하다면 개인 cpu의 코어수만큼 NUM_WORKERS를 설정하면 최대의 데이터 수집속도를
 경험할 수 있습니다.


---

## 평가
<br />

 iteration을 반복할 때마다 이전 iteration에서의 모델과 21게임을 진행하여 승률을
 측정합니다.
 <br />
 <br />

 * iteration_0의 평가
     - **vs Mint model** -> Win : 21.00  Draw : 0.00  Lose : 0.00 (over 21 games)
     <br />

 * iteration_1의 평가
     - **vs Mint model** -> Win : 21.00  Draw : 0.00  Lose : 0.00 (over 21 games)
     <br />

     - **vs iter_00 model**  -> Win : 16.00  Draw : 0.00  Lose : 5.00 (over 21 games)
     <br />

 * iteration_2의 평가
     - **vs Mint model** -> Win : 21.00  Draw : 0.00  Lose : 0.00 (over 21 games)
     <br />

     - **vs iter_00 model**  -> Win : 21.00  Draw : 0.00  Lose : 0.00 (over 21 games)
     <br />

     - **vs iter_01 model**  -> Win : 14.00  Draw : 1.00  Lose : 15.00 (over 30 games)
     <br />

 iter_2 모델이 iter_1 모델과의 게임에서 특별히 강한 모습을 보여주지 않아서 조기에 학습을
 종료하였습니다. 하지만 iter_0과 게임하는 것에 대해서는 iter_2가 iter_1보다 확실히 더
 강한 모습을 보이는 것을 보면 iter_2 모델에서 더 강화학습을 진행하면 더 강한 모델을 얻을
 수 있을 수 있을 것이라고 생각합니다.
<br />
<br />

 * **테스트 경기**
  iter_02 모델과 iter_00 모델의 테스트 경기 중 하나의 기보입니다. X(백)에 해당하는
 Net 1이 iter_02 모델, O(흑)에 해당하는 Net 2가 iter_00의 모델입니다.

     - **Visit** : 해당 액션에 해당하는 노드를 방문한 횟수입니다. 이 횟수가 높을 수록
       이 행동을 택할 확률이 높습니다.
       <br />

     - **Q_Value** : 해당 액션의 가치를 추정한 값입니다. 1에 가까우면 자신에게 유리, -1에
       가까우면 자신에게 불리하다고 판단합니다.
       <br />

     - **Prob** : 해당 액션을 택할 확률을 나타냅니다.
       <br />


![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game01.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game02.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game03.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game04.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game05.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game06.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game07.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game08.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game09.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game10.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game11.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game12.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game13.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game14.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game15.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game16.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game17.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game18.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game19.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game20.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game21.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game22.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game23.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game24.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game25.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game26.jpg)
![game record]({{ site.baseurl }}/images/AIX_Project_Part04/game27.jpg)




---
***

완성된 최종 파일은 [여기](https://github.com/hissmell/Pytorch_Toy_Projects/tree/main/Omok)에서 확인할 수 있습니다.

---
***