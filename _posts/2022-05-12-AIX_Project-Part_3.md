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
 ![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/CNN01.png "Example of Omok game")
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

![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/batchnorm01.png "Example of Omok game")
<br />
<br />
 신경망을 학습할때, 학습시키는 데이터의 값들의 분포가 치우쳐져 있지 않고 분산이 적을 수록
 학습이 원활하게 진행되는 경향이 있습니다. batch normalization 레이어는 학습 속도를
 높이기 위해서, 인풋데이터들을 평균이 0, 분산이 1이 되도록 값들을 재조정하는 역할을 수행합니다.
<br />
<br />
![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/batchnorm02.png "Example of Omok game")
<br />
<br />
 batch normalization layer를 거친 데이터 값들이 그렇지 않은 값들에 비해서 평균이 0에 가까운
 고른 분포를 가진 것을 확인 할 수 있습니다.

---

## **Residual Network**
<br />

 ![an image alt text]({{ site.baseurl }}/images/AIX_Project_Part03/residual01.png "Example of Omok game")
<br />
<br />
 신경망이 복잡한 상황에 대한 충분한 판단능력을 가지기 위해서는 신경망을 깊게 쌓는것이
 필요합니다. 하지만 신경망이 깊어지면 깊어질 수록, 역전파과정에서 전달되는 gradient가
 소실되는 "**gradient vanishing**"문제가 필연적으로 발생합니다.*"gradient vanishing"**문제가 필연적으로 발생합니다.  
 이러한 문제를 해결하기 위해서 residual network는 들어오는 인풋을 2개의 트랙으로 나누어
 한 트랙은 일반적인 활성화 함수를 사용하고, 다른 한 트랙은 gradient가 그대로 전달되도록
 더하기 연산만 진행하는 전략을 사용합니다.  
 이러한 개선 덕분에 residual network는 신경망을 깊게 하더라도 gradient vanishing
 문제가 발생하지 않는 모습을 보이며, 학습의 속도도 개선 이전보다 더 빠른 모습을 보입니다.

