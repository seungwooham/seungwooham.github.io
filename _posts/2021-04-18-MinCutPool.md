---
title: "MinCutPool"
date: 2021-04-18
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

$$\bar{\mathbf{X}}$$는 MP(Message passing, 하나의 graph neural network로 생각하면 됨)들을 통과하여 얻어진 node representation matrix입니다. $$\bar{\mathbf{X}}$$를 multi-layer perceptron(MLP, basic한 인공 신경망)과 softmax를 거치게 하여 cluster assignment $${\mathbf{S}}$$를 얻습니다. 이때 $$\Theta_{\mathrm{GNN}}$$와 $$\Theta_{\mathrm{MLP}}$$는 훈련 가능한 파라미터입니다. softmax를 통과하며 $$\mathbf{S}$$의 원소는 $$\mathbf{s}_{ij}\in [0,1]$$를 만족합니다. 그리고 $$\mathbf{S}\mathbf{1}_K=\mathbf{1}_N$$가 성립하게됩니다. ($$[N \times K] \times [K \times 1]=[N \times 1]$$, N개의 node, K개의 cluster)

$$\begin{aligned}
\mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{X} \\ 
\mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
\end{aligned}$$


### 출처
Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>
Han, Yufei, and Maurizio Filippone. "Mini-batch spectral clustering." In 2017 International Joint Conference on Neural Networks (IJCNN), pp. 3888-3895. IEEE, 2017. <br/>
Tian, Fei, Bin Gao, Qing Cui, Enhong Chen, and Tie-Yan Liu. "Learning deep representations for graph clustering." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 28, no. 1. 2014. <br/>