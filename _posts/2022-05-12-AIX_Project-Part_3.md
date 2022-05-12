---
layout : post
title : Part 3. 신경망
---



이 포스트에서는 프로젝트에 사용된 신경망에 대해서 간략히 설명합니다.

---

## **Convolutional layer**
<br />


![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/Outline-of-the-convolutional-layer.png "Example of Omok game")
<br />
<br />
 Convolutional layer는 이미지형식의 파일을 인풋으로 받아서 자신이 가지고 있는 필터를
 사용해 합성곱 연산을 진행합니다.
 Convolutional layer는 단순히 행렬곱 연산을 진행하는 Fully Connected layer와는
  다르게 **"국소적"인 특징을 이해할 수 있다**는 특징을 가지고 있습니다.
<br />
<br />
 ![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/CNN01.png)
<br />
<br />
 따라서 Convolutional layer를 여러 층 쌓게 된다면 초기의 레이어들은 저수준의 국소적인
 특징을 알아내고, 깊은 레이어일 수록 고수준의 특징을 구별할 수 있게 됩니다.  
 Convolutional layer는 이미지의 분류, 영상 인식과 같은 분야에서 주로 이용됩니다.

<br />
<br />

---

## **Batch normalization layer**
<br />

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/batchnorm01.jpg "Example of Omok game")
<br />
<br />
 신경망을 학습할때, 학습시키는 데이터의 값들의 분포가 치우쳐져 있지 않고 분산이 적을 수록
 학습이 원활하게 진행되는 경향이 있습니다. batch normalization 레이어는 학습 속도를
 높이기 위해서, 인풋데이터들을 평균이 0, 분산이 1이 되도록 값들을 재조정하는 역할을 수행합니다.
<br />
<br />
![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/batchnorm02.jpg "Example of Omok game")
<br />
<br />
 batch normalization layer를 거친 데이터 값들이 그렇지 않은 값들에 비해서 평균이 0에 가까운
 고른 분포를 가진 것을 확인 할 수 있습니다.

<br />
<br />

---

## **Residual Network**
<br />

 ![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/residual01.png "Example of Omok game")
<br />
<br />
 신경망이 복잡한 상황에 대한 충분한 판단능력을 가지기 위해서는 신경망을 깊게 쌓는것이
 필요합니다. 하지만 신경망이 깊어지면 깊어질 수록, 역전파과정에서 전달되는 gradient가
 소실되는 "**gradient vanishing**"문제가 필연적으로 발생합니다.*"gradient vanishing"**문제가 필연적으로 발생합니다.
<br />
<br />
 이러한 문제를 해결하기 위해서 residual network는 들어오는 인풋을 2개의 트랙으로 나누어
 한 트랙은 일반적인 활성화 함수를 사용하고, 다른 한 트랙은 gradient가 그대로 전달되도록
 더하기 연산만 진행하는 전략을 사용합니다.
<br />
<br />
 이러한 개선 덕분에 residual network는 신경망을 깊게 하더라도 gradient vanishing
 문제가 발생하지 않는 모습을 보이며, 학습의 속도도 개선 이전보다 더 빠른 모습을 보입니다.

<br />
<br />

---

## **Building neural network with PyTorch**
<br />

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/network.PNG)
<br />
<br />
 전체 네트워크 구조는 위 그림과 같습니다.
<br />
<br />

* **Convolutional Block**
<br />

처음 State와 만나는 레이어입니다. State의 국소적 특징을 추출하고, Residual tower에
들어가기 좋은 shape으로 변형합니다.
<br />
<br />

```python
class ConvolutionalBlock(nn.Module):
    def __init__(self,inchannel_num,filter_num = 256):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(inchannel_num,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=filter_num)
        self.relu = nn.ReLU()

    def forward(self,input):
        x = self.conv(input)
        x = self.batch_norm(x)
        output = self.relu(x)
        return output
```
<br />
<br />

* **Residual Tower**
<br />

Residual block을 여러게 쌓은 레이어입니다. 사실상 대부분의 연산을 담당합니다.
<br />
<br />
```python
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,filter_num=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(filter_num)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filter_num,filter_num,kernel_size=3,stride=1,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(filter_num)
        self.relu2 = nn.ReLU()

    def forward(self,input):
        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.add(x,input)
        x = self.relu2(x)
        return x
        
# Residusal Tower는
# nn.Sequential(*[ResidualBlock(256) for _ in range(19)])
# 의 형태로 사용합니다.
```
<br />
<br />

* **전체 신경망**
<br />

policy head와 value head는 내부에 구현되어 있습니다.
<br />
<br />
```python
class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.convolutional_block = ConvolutionalBlock(2)
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(256) for _ in range(19)
        ])

        #policy head
        self.p_conv = nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0)
        self.p_batch_norm = nn.BatchNorm2d(2)
        self.p_relu = nn.ReLU()
        policy_conv_out = self._get_policy_conv_out()
        self.p_dense = nn.Linear(policy_conv_out,81)

        #value head
        self.v_conv = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.v_batch_norm1 = nn.BatchNorm2d(1)
        self.v_relu1 = nn.ReLU()
        value_conv_out = self._get_value_conv_out()
        self.v_dense1 = nn.Linear(value_conv_out,256)
        self.v_relu2 = nn.ReLU()
        self.v_dense2 = nn.Linear(256,1)
        self.v_tanh = nn.Tanh()

    def forward(self,input):
        x = self.convolutional_block(input)
        x = self.residual_tower(x)

        p = self.p_conv(x)
        p = self.p_batch_norm(p)
        p = self.p_relu(p)
        p = self.p_dense(p.view(p.size()[0],-1))

        v = self.v_conv(x)
        v = self.v_batch_norm1(v)
        v = self.v_relu1(v)
        v = self.v_dense1(v.view(v.size()[0],-1))
        v = self.v_relu2(v)
        v = self.v_dense2(v)
        v = self.v_tanh(v)
        return p,v

    def _get_policy_conv_out(self):
        temp = torch.zeros(1,2,9,9)
        temp = self.convolutional_block(temp)
        temp = self.residual_tower(temp)
        temp = self.p_conv(temp)
        temp = self.p_batch_norm(temp)
        temp = self.p_relu(temp)
        return int(np.prod(temp.size()))

    def _get_value_conv_out(self):
        temp = torch.zeros(1,2,9,9)
        temp = self.convolutional_block(temp)
        temp = self.residual_tower(temp)
        temp = self.v_conv(temp)
        temp = self.v_batch_norm1(temp)
        temp = self.v_relu1(temp)
        return int(np.prod(temp.size()))
```
<br />
<br />

---
***

완성된 신경망은 [여기](https://github.com/hissmell/Pytorch_Toy_Projects/blob/Omok/Omok/lib/models.py)
서 확인 할 수 있습니다.

---
***

