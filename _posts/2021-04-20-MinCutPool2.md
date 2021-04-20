---
title: "MinCutPool: 이론편2 (MinCutPool: Understanding the Theory 2)"
date: 2021-04-20
categories:
  - Machine Learning
tags:
  - 한글
---
### MinCutPool: 이론편2 (MinCutPool: Understanding the Theory 2)

이론편1은 [여기](<https://seungwooham.github.io/machine%20learning/MinCutPool1/>)에서 확인하실 수 있습니다.

지난 1편에서 local minima에 빠지기 쉬운 cut loss를 보안하기 위해 orthogonality loss가 있다는 이야기까지 했습니다. Cut loss가 빠질 수 있는 local minima에는 두 가지가 있었는데, 첫 번째는 모든 node가 모든 cluster에 같은 정도로 속하는 것, 두 번째는 모든 node가 한 cluster에만 속하는 경우였습니다. Orthogonality loss는 이 문제를 해결하기 위해 cluster assignment가 orthogonal 하도록(모든 node가 모든 cluster에 같은 정도로 속하는 경우 해결), 그리고 cluster의 size가 비슷하도록(모든 node가 한 cluster에만 속하는 경우 해결) 유도합니다. 식은 아래와 같이 나타납니다.

$$\begin{aligned}
\mathcal{L}_o = { \left\| \frac{\mathbf{S}^{\top} \mathbf{S}}{ { \| \mathbf{S}^{\top} \mathbf{S} \| }_F } - \frac{\mathbf{I}_C}{\sqrt{C}} \right\| }_F
\end{aligned}$$

여기서 아래 첨자 F는 Frobenius norm의 약자로 $${ \left\| \mathbf{X} \right\| }_F$$는 $$\mathbf{X}$$의 Frobenius norm이라는 의미입니다. Frobenius norm은 다음과 같이 정의됩니다.

$$\begin{aligned}
{ \left\| \mathbf{A} \right\| }_F = \sqrt{ \sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^{2} } = \sqrt{ \mathrm{trace}(\mathbf{A} * \mathbf{A}) } = \sqrt { \sum_{i=1}^{min \{ m,n \} } \sigma_{i}^{2} (\mathbf{A}) }
\end{aligned}$$

이때 $$\sigma_{i}^{\mathbf{A}}$$는 $$\mathbf{A}$$의 singular value를 의미합니다. 

```python
import numpy as np

# 최적 cluster
c = [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]

# 예측 cluster
s = [[0.8, 0.2, 0.0],
     [0.8, 0.2, 0.0],
     [0.8, 0.2, 0.0],
     [0.2, 0.6, 0.2],
     [0.2, 0.6, 0.2],
     [0.2, 0.6, 0.2],
     [0.0, 0.1, 0.9],
     [0.0, 0.1, 0.9],
     [0.0, 0.1, 0.9]]

# 극도로 편향된 cluster
ext = [[1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0],
       [1, 0, 0]]

def lo1(mat):
  numerator = np.transpose(mat) @ mat
  denominator = np.sqrt(np.sum(np.square(np.transpose(mat) @ mat)))
  return numerator/denominator

def loss(lo1, lo2):
  loss = np.sqrt(np.sum(np.square(lo1+lo2)))
  return loss

lo2 = -np.identity(3)/np.sqrt(3)

print('최적 cluster의 orthogonality loss = {:.4f}'.format(loss(lo1(c), lo2)))
print('예측 cluster의 orthogonality loss = {:.4f}'.format(loss(lo1(s), lo2)))
print('극단적인 cluster의 orthogonality loss = {:.4f}'.format(loss(lo1(ext), lo2)))

'''
최적 cluster의 orthogonality loss = 0.0000
예측 cluster의 orthogonality loss = 0.4793
극단적인 cluster의 orthogonality loss = 0.9194
'''
```



### 출처
- Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>