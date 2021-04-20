---
title: "MinCutPool"
date: 2021-04-19
categories:
  - Machine Learning
tags:
  - 한글
---
### MinCutPool

MinCutPool은 복잡한 그래프에서 주요한 node를 추출하고자 한 여러 시도들 중 하나입니다. 일반적으로 이런 node 추출은 graph를 matrix로 표현한 뒤, 고유값 분해를 통해 핵심 node를 결정합니다. 하지만 이런 고유값 분해 과정은 계산 복잡도가 높아 ($$O(N^3)$$) scalability issue가 있습니다. 최근 gradient descentt 알고리즘을 활용하여 복잡도를 $$O(N^2)$$이나 $$O(N)$$까지 줄이는 연구가 있었습니다(Han & Filippone, 2017). 인공신경망을 이용한 연구도 진행되어 Autoencoder를 활용해 Laplacian matrix(Degree matrix - Adjacency matrix)의 i번째 행을 주요한 eigenvector들의 i번째 component에 연결시키는 작업도 이루어졌습니다(Tian et al., 2014). [(Laplacian matrix에 대한 설명은 이 링크를 클릭하세요.)](<https://junklee.tistory.com/112/>)

본 논문에서는 spectral clustering(SC, eigenvector 기반 clustering)의 단점을 해소하면서 graph topology와 node의 feature를 기반으로 node를 clustering하는 방법론을 제시합니다. 아래는 자세한 방법입니다.

$$\begin{aligned}
\bar{\mathbf{X}} &= \mathrm{GNN}(\mathbf{X},\tilde{\mathbf{A}};\mathbf{\Theta_{\mathrm{GNN}}}) \\
\mathbf{S} &= \mathrm{softmax}(\mathrm{MLP}(\bar{\mathbf{X}};\Theta_{\mathrm{MLP}}))
\end{aligned}$$

$$\bar{\mathbf{X}}$$는 MP(Message passing, 하나의 graph neural layer 층으로 생각하면 됨)들을 통과하여 얻어진 node representation matrix입니다. $$\bar{\mathbf{X}}$$를 multi-layer perceptron(MLP, basic한 인공 신경망)과 softmax에 연달아 통과시켜 cluster assignment $${\mathbf{S}}$$를 얻습니다. 이때 $$\Theta_{\mathrm{GNN}}$$와 $$\Theta_{\mathrm{MLP}}$$는 훈련 가능한 파라미터입니다. $$\mathbf{S}$$는 $$[N \times K]$$ 행렬으로 i번째 행에 node i가 각 cluster에 속할 확률을 담고 있습니다. $$\mathbf{S}$$가 softmax를 통과한 결과물이기 때문에 $$\mathbf{S}$$의 원소 $$\mathbf{s}_{ij}$$는 $$\mathbf{s}_{ij}\in [0,1]$$를 만족합니다. 그리고 $$\mathbf{S}\mathbf{1}_K=\mathbf{1}_N$$가 성립하게됩니다. ($$[N \times K] \times [K \times 1]=[N \times 1]$$, N개의 node, K개의 cluster)

그래프의 예시와 함께 확인해보겠습니다. 아래는 9개의 node로 이루어진 그래프와 그 인접행렬입니다.

![Figure_1](/assets/images/2021-04-19-MCP1.png)

이 그래프의 최적 클러스터가 각각 적색, 녹색, 청색이라고 가정해봅시다. 그러면 최적 cluster 구성을 담은 행렬 $$\mathbf{C}$$를 표기할 수 있습니다. 그리고 $$\mathbf{C}\mathbf{1}_K=\mathbf{1}_N$$가 됨을 보일 수 있습니다.

![Figure_2](/assets/images/2021-04-19-MCP2.png)

최적 cluster(정답)가 아닌 예측 결과 행렬 $$\mathbf{S}}$$에서도 $$\mathbf{S}}\mathbf{1}_K=\mathbf{1}_N$$는 성립합니다. $$\mathbf{S}}$$의 첫 번째 행은 첫 번째 node가 각 cluster에 속할 확률을 나타냅니다.

![Figure_3](/assets/images/2021-04-19-MCP3.png)

행렬 $$\mathbf{S}$$는 MinCutPool의 loss에 중요하게 작용합니다. MinCutPool은 두 가지 loss로 이루어져 있습니다. 첫 번째는 cut loss $$\mathcal{L}_c$$, 두 번째는 orthogonality loss $$\mathcal{L}_o$$입니다. $$\mathcal{L}_c$$에 대해서 먼저 확인해보겠습니다.

$$\begin{aligned}
\mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A} \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D} \mathbf{S})}
\end{aligned}$$

$$\mathcal{L}_c$$는 'cluster 내부의 연결성/graph 전체의 연결성'으로, cluster 내부가 전체적인 graph의 연결성과 비교해서 얼마나 끈끈히 연결되어 있는지 알려줍니다. $$\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A} \mathbf{S})$$이 cluster 내부의 연결성, $$\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}} \mathbf{S})$$가 graph 전체의 연결성을 나타냅니다. 예시 code와 함께 살펴봅시다.

```python
import numpy as np
c = [[1, 0, 0],
     [1, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]

s = [[0.8, 0.2, 0.0],
     [0.8, 0.2, 0.0],
     [0.8, 0.2, 0.0],
     [0.2, 0.6, 0.0],
     [0.2, 0.6, 0.0],
     [0.2, 0.6, 0.0],
     [0.0, 0.1, 0.9],
     [0.0, 0.1, 0.9],
     [0.0, 0.1, 0.9]]

A = [[0 ,1, 1, 0, 0, 0, 0, 0, 0],
     [1 ,0, 1, 0, 0, 0, 0, 0, 0],
     [1 ,1, 0, 0, 0, 1, 0, 0, 1],
     [0 ,0, 0, 0, 1, 1, 0, 0, 0],
     [0 ,0, 0, 1, 0, 1, 0, 0, 0],
     [0 ,0, 1, 1, 1, 0, 0, 0, 1],
     [0 ,0, 0, 0, 0, 0, 0, 1, 1],
     [0 ,0, 0, 0, 0, 0, 1, 0, 1],
     [0 ,0, 1, 0, 0, 1, 1, 1, 0]]

print(np.transpose(c) @ A)
'''
array([[2, 2, 2, 0, 0, 1, 0, 0, 1],
       [0, 0, 1, 2, 2, 2, 0, 0, 1],
       [0, 0, 1, 0, 0, 1, 2, 2, 2]])
i번째 행, j 번째 열의 원소는 i 번째 cluster와 j 번째 node 사이의 edge 수를 나타냅니다.
'''

print(np.transpose(c) @ A @ c)
'''
array([[6, 1, 1],
       [1, 6, 1],
       [1, 1, 6]])
i번째 행, j 번째 열의 원소는 i 번째 cluster와 j 번째 cluster 사이의 edge 수를 나타냅니다.
'''

D = np.diag(np.sum(A, axis=1) # Degree matrix를 생성합니다.

np.transpose(s) @ D
# 마치 연결의 가중치를 분배하는 느낌. 총 4개의 연결이 있다면 이 연결 중에서 cluster 몇에 할당할 연결은 얼마다. 

```

$$\begin{aligned}
\mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{X} \\ 
\mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
\end{aligned}$$


### 출처
Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>
Han, Yufei, and Maurizio Filippone. "Mini-batch spectral clustering." In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 3888-3895. IEEE, 2017. <br/>
Tian, Fei, Bin Gao, Qing Cui, Enhong Chen, and Tie-Yan Liu. "Learning deep representations for graph clustering." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 28, no. 1. 2014. <br/>