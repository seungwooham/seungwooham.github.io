---
title: "CS224w - 17. Scaling Up Graph Neural Networks Part 2"
date: 2022-12-05 15:56:42 +0900
categories: [ML/DL, CS224W]
tags: [Korean, CS224W, GNN, large-graphs, scalability]
original_url: https://seungwooham.tistory.com/88
math: true
---

## Advanced Cluster-GCN

![](/assets/img/posts/88/01.jpg){: width="3000" height="2250" }

Advanced Cluster-GCN의 아이디어는 몇개의 node group을 mini-batch에서 합치는 것입니다. 이를 위해서 조금 더 작은 node group을 만들게 됩니다.

![](/assets/img/posts/88/02.jpg){: width="3000" height="2250" }

여러 노드 그룹을 쓰기 때문에 전체 그래프를 보다 잘 표현할 수 있습니다. 그리고 그룹 간의 링크가 있어, message가 그룹 사이에도 흐를 수 있습니다.

![](/assets/img/posts/88/03.jpg){: width="3000" height="2250" }

Pre-processing 단계에서는, vanilla와 같이 subgraph로 나누되 더 작게 나눠서 나중에 aggregate 됐을 때 사이즈가 너무 커지지 않게 합니다.

![](/assets/img/posts/88/04.jpg){: width="3000" height="2250" }

Mini-batch training은 subgraph 중 random으로 일부 샘플링합니다. 그다음 샘플링한 subgraph들에 속한 node들을 전부 aggregate해서 하나의 subgraph로 만듭니다. 이때 그룹 사이의, between-group edge를 포함시킵니다.

![](/assets/img/posts/88/05.jpg){: width="3000" height="2250" }

Time complexity를 neighbor sampling과 비교해보면, neighbor sampling은 $$MH^K$$입니다.

![](/assets/img/posts/88/06.jpg){: width="3000" height="2250" }

이와 비교하여 Cluster-GCN은 $$KMD_{avg}$$의 복잡도를 갖습니다.

![](/assets/img/posts/88/07.jpg){: width="3000" height="2250" }

이웃 노드 절반만 샘플링해서 $$H=D_{avg}/2$$ 라고 할 때, Cluster-GCN은 $$2MHK$$, neighbor sampling은 $$MH^K$$ 의 time complexity를 가지며, Cluster-GCN은 K에 대한 linear한 dependency를 가집니다.

![](/assets/img/posts/88/08.jpg){: width="3000" height="2250" }

요약하면 아래와 같습니다.

- Cluster-GCN은 전체 노드를 작은 노드 그룹으로 나누고 이를 mini-batch에서 합쳐 학습
- Layer-wise node embedding을 하기 때문에 layer 수가 많을 때 특히 neighbor sampling 보다 효율적
- 하지만 cross-community edge를 상실해 학습이 편향된다는 단점 존재

## Simplifying GNN Architecture

![](/assets/img/posts/88/09.jpg){: width="3000" height="2250" }

GCN에서 non-linear activation을 없앤 Simplified GCN(Wu et al., ICML 2019)은 benchmark 성능이 크게 떨어지지 않으면서도, 모델 구조 덕분에 scalability가 매우 높습니다. 이 단순화 방식은 앞서 recommender system에서 본 LightGCN과 매우 비슷합니다. 먼저 LightGCN에 대해서 확인해보겠습니다.

![](/assets/img/posts/88/10.jpg){: width="3000" height="2250" }

먼저 LightGCN은 ReLU non-linearity가 없습니다.

![](/assets/img/posts/88/11.jpg){: width="3000" height="2250" }

Simplified GCN에는 LightGCN과 달리 self-loop가 있고, Input embedding $$E$$가 학습되는 feature라기보다 given feature에 가깝습니다. $$\tilde{A}^KE$$는 전처리 과정에 가까워서 한 번만 계산되면 됩니다.

![](/assets/img/posts/88/12.jpg){: width="3000" height="2250" }

Simplified GCN은 전처리된 feature $$\tilde{A}^KE$$ 위에서 linear model만 학습시켜도 GCN과 비슷한 성능을 낸다는 것을 보여주었습니다. 다만 non-linearity가 없어 표현력이 떨어집니다. 하지만 semi-supervised node classification에서 simplified GCN은 original GNN과 거의 비슷한 성능을 보였습니다. 그 원리가 무엇일까요?

![](/assets/img/posts/88/13.jpg){: width="3000" height="2250" }

많은 node classification에서는 homophily structure가 관찰되었습니다. 따라서 연결된 노드들은 비슷한 라벨을 가질 가능성이 큽니다.

![](/assets/img/posts/88/14.jpg){: width="3000" height="2250" }

Simplified GCN은 neighbor node feature를 반복적으로 평균내는 것이기 때문에, 연결되어 있는 노드들은 비슷한 pre-processed feature를 갖게 됩니다.

![](/assets/img/posts/88/15.jpg){: width="3000" height="2250" }

그래서 연결되어 있는 노드들은 모델에 의해 같은 라벨로 예측되게 되고, Simplified GCN은 graph homophily를 잘 반영하는 모델인 것입니다. 이는 많은 node classification dataset에 적합한 방법입니다.
