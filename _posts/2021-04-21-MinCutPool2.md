---
title: "MinCutPool: 이론편2 (MinCutPool: Understanding the Theory 2)"
date: 2021-04-21
categories:
  - Machine Learning
tags:
  - 한글
---
## MinCutPool: 이론편2 (MinCutPool: Understanding the Theory 2)

이론편1은 [여기](<https://seungwooham.github.io/machine%20learning/MinCutPool1/>)에서 확인하실 수 있습니다.

지난 1편에서 local minima에 빠지기 쉬운 cut loss를 보안하기 위해 orthogonality loss가 있다는 이야기까지 했습니다. Cut loss가 빠질 수 있는 local minima에는 두 가지가 있었는데, 첫 번째는 모든 node가 모든 cluster에 같은 정도로 속하는 것, 두 번째는 모든 node가 한 cluster에만 속하는 경우였습니다. Orthogonality loss는 이 문제를 해결하기 위해 cluster assignment가 orthogonal 하도록(모든 node가 모든 cluster에 같은 정도로 속하는 경우 해결), 그리고 cluster의 size가 비슷하도록(모든 node가 한 cluster에만 속하는 경우 해결) 유도합니다. 식은 아래와 같이 나타납니다.

$$\begin{aligned}
\mathcal{L}_o = { \left\| \frac{\mathbf{S}^{\top} \mathbf{S}}{ { \| \mathbf{S}^{\top} \mathbf{S} \| }_F } - \frac{\mathbf{I}_C}{\sqrt{C}} \right\| }_F
\end{aligned}$$

여기서 아래 첨자 F는 Frobenius norm의 약자로 $${ \left\| \mathbf{X} \right\| }_F$$는 $$\mathbf{X}$$의 Frobenius norm이라는 의미입니다. Frobenius norm은 다음과 같이 정의됩니다.

$$\begin{aligned}
{ \left\| \mathbf{A} \right\| }_F = \sqrt{ \sum_{i=1}^{m} \sum_{j=1}^{n} \vert a_{ij} \vert ^{2} } = \sqrt{ \mathrm{trace}(\mathbf{A} * \mathbf{A}) } = \sqrt { \sum_{i=1}^{min \{ m,n \} } \sigma_{i}^{2} (\mathbf{A}) }
\end{aligned}$$

이때 $$\sigma_{i}(\mathbf{A})$$는 $$\mathbf{A}$$의 singular value를 의미합니다. 실제 code 작성할 때에는 첫 번째 정의인 $$\sqrt{ \sum_{i=1}^{m} \sum_{j=1}^{n} {\vert a_{ij} \vert} ^{2}}$$를 활용하였습니다. 몇 가지 예시와 함께 결과를 살펴봅시다.

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

완벽하게 분배된 cluster의 경우 orthogonality loss가 0이 되는 것을 확인할 수 있습니다. Cluster assignment가 오차 범위를 넘어 극단으로 갈수록 loss는 1에 가까워집니다. $$\frac{\mathbf{I}_C}{\sqrt{C}}$$은 모든 원소의 값이 $$1/\sqrt{C}$$이고 Frobenius norm이 1입니다. $$\frac{\mathbf{S}^{\top} \mathbf{S}}{ { \| \mathbf{S}^{\top} \mathbf{S} \| }_F }$$ 또한 Frobenius norm이 1이므로, 원소가 최대한 균등하게 퍼져있으면서(cluster에 속한 node수가 유사함) 동시에 대각성분에만 집중되어있는 것(cluster가 서로 orthogonal함)이 orthogonality loss를 줄일 수 있습니다.

전통적인 spectral clustering(SC) 방법론은 매 sample마다 spectral decomposition을 진행해야 했습니다. MinCutPool에서는 cluster assignment가 인공 신경망에 의해 계산됩니다. 인공 신경망은 node feature space에서 cluster assignment space로 mapping하는 함수 역할을 합니다. 인공 신경망의 파라미터가 graph의 크기에 독립이고, GNN의 message passing(MP) operation이 node space에서만 작동하므로(Laplacian의 spectrum과 무관) MinCutPool은 여러 graph에 일반적으로 적용될 수 있습니다. 이 때문에 작은 규모의 graph에서 훈련을 하고 큰 graph에서 cluster하는 방법도 가능합니다. (방금 두 문장은 아직 이해하지 못했습니다.)

지금까지는 loss에 대한 이해였습니다. Cut loss와 orthogonality loss의 조합은 다른 cluster 문제에도 적용할 수 있습니다. 이제 알고리즘 파트로 넘어갑니다. 기본적으로 MinCutPool은 다음과 같은 과정으로 진행됩니다.

$$\begin{aligned}
\mathbf{X}^{pool} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{X} \\
\mathbf{A}^{pool} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
\end{aligned}$$

$$\mathbf{X}^{pool} \in \mathbb{R}^{K \times F}$$에 속한 $$x^{pool}_{ij}$$는 cluster i의 elements를 feature j에 따라 합한 값입니다. 대칭 행렬 $$\mathbf{A}^{pool} \in \mathbb{R}^{K \times K}$$에 속한 $$a^{pool}_{ii}$$는 cluster i에 속한 edge 내부의 weighted sum 입니다. $$a^{pool}_{ij}$$는 cluster i와 cluster j 사이의 weighted sum 입니다. $$\mathbf{A}^{pool}$$은 cut loss를 거치면서 trace의 값을 최대화 하도록 수정되기 때문에, 대각 성분이 주요한 원소가 되는(cluster 내부의 연결이 중요해지는) 행렬로 변화합니다. 대각 성분에만 값이 집중된 인접 행렬(adjacency matrix)는 self-loop를 만들어 MP operation의 다른 node로의 전파를 방해합니다. 따라서 본 연구에서는 대각 성분을 0으로 만든 후 degree normalization을 거친 새로운 adjacency matrix를 제안합니다. 

$$\begin{aligned}
& \hat{\mathbf{A}} = \mathbf{A}^{pool}-\mathbf{I}_K \mathrm{diag}(\mathbf{A}^{pool}) \\
& \tilde{\mathbf{A}}^{pool} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}
\end{aligned}$$

식을 보면, 자기 연결성을 아예 제외하고 다시 한 번 degree normalization을 도입합니다. 이 부분은 기존의 GNN을 약간 변형한 것으로 보시면 되겠습니다. 저자는 MinCutPool의 특징을 후반부에 다시 정리합니다.

1. Node feature는 MP operation 과정에서 서로 유사해집니다. MinCutPool은 유사한 node feature들을 기반으로 clustering을 진행합니다. 이 과정에서 MinCutPool의 결과 cluster는 서로 강하게 연결되어 있으면서, 유사한 feature를 갖는 node를 연결하게 됩니다. 그리고 degenerate 해 (모든 node가 전체 cluster에 속하거나, 모든 node가 오직 하나의 cluster에만 속하거나)를 피합니다.
2. $$\mathcal{L}_c$$의 degenerate minima(옳지 않은 minima)는 graph의 대부분의 정보를 소멸시킵니다. 하지만 orthogonality loss의 등장으로 해당 문제가 해결됩니다.

MinCutPool의 space complexity는 $$\mathcal{O}(NK)$$입니다. Node 개수가 많을수록, cluster 수가 많을수록 증가하게 됩니다. Computational complexity는 $$\mathcal{L}_c$$의 분자인 $$\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A} \mathbf{S})$$에 의해 결정됩니다. 계산된 computational complexity는 $$\mathcal{O}(NK(N+K))$$ 입니다. 하지만 $$\mathbf{A}^{pool}$$가 주로 sparse한 matrix이기 때문에 $$\mathcal{O}(K(E+NK))$$까지 낮아질 수 있다고 합니다. 이제 이론은 충분히 살펴보았으니, 코드와 함께 분석을 진행해봅시다.

### 출처
- Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>