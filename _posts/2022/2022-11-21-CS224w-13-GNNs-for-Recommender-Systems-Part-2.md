---
title: "CS224w - 13. GNNs for Recommender Systems Part 2"
date: 2022-11-21 17:14:00 +0900
categories: [ML/DL, CS224W]
tags: [Korean, CS224W, GNN, recommender-system]
original_url: https://seungwooham.tistory.com/80
math: true
---

## LightGCN

![](/assets/img/posts/80/01.webp){: width="3000" height="2250" }

NGCF에서 학습하는 파라미터는 두 가지로, Shallow user/item embeddings, GNN parameter입니다. 하지만 굳이 GNN이 끼지 않아도 shallow learnable embedding은 노드 단위로 이루어지기 때문에 충분히 expressive하다고 볼 수 있습니다.

![](/assets/img/posts/80/02.webp){: width="3000" height="2250" }

LightGCN은 NGCF를 세 가지 방법으로 simplify합니다. Bipartite graph의 adjacency matrix 정의, GCN의 matrix formulation, 그리고 non-linearity 제거를 통한 GCN 단순화로 오히려 recommendation performance가 더 향상되는 결과를 보였습니다.

![](/assets/img/posts/80/03.webp){: width="3000" height="2250" }

먼저 user와 item에 대해서 위와 같은 새로운 형태의 adjacency matrix를 제시할 수 있습니다. User와 item이 개별 축에 존재하던 형태와는 조금 다른 adjacency matrix입니다.

![](/assets/img/posts/80/04.webp){: width="3000" height="2250" }

기존의 GCN은 위와 같이 aggregation과 learnable linear transformation으로 이루어져 있었습니다. 이 또한 단순화가 가능합니다.

![](/assets/img/posts/80/05.webp){: width="3000" height="2250" }

먼저 ReLU를 제거하여 전체 표현형을 매우 간단하게 만들 수 있습니다. $$\widetilde{A}^{K}$$와 $${W}^{(0)} \cdots {W}^{(K-1)}$$으로 이루어진 표현형이 가능합니다.

![](/assets/img/posts/80/06.webp){: width="3000" height="2250" }

정리하면 위의 형태로 단순화됩니다.

![](/assets/img/posts/80/07.webp){: width="3000" height="2250" }

Multi-scale diffusion은 over-smoothing을 방지하기 위해 각 layer의 $$E^{(k)}$$에 가중치 $$\alpha$$ 를 곱해 더하는 것입니다. 모델의 단순화를 위해 LightGCN에서는 uniform한 $$\alpha$$를 사용했습니다.

![](/assets/img/posts/80/08.webp){: width="3000" height="2250" }

위의 형태대로 $$E^{(k+1)}$$이 얻어집니다.

![](/assets/img/posts/80/09.webp){: width="3000" height="2250" }

마지막 final embedding은 각 embedding에 $$1/(K+1)$$을 곱해서 더한 것입니다.

![](/assets/img/posts/80/10.webp){: width="3000" height="2250" }

이때 얻어진 final embedding에서 user와 item의 embedding을 얻어 내적 하면 score를 계산할 수 있습니다.

![](/assets/img/posts/80/11.webp){: width="3000" height="2250" }

이렇게 단순한 diffusion propagation이 작동하는 이유는 diffusion이 similar user/item의 embedding이 similar 하도록 유도하기 때문으로 보고 있습니다.

![](/assets/img/posts/80/12.webp){: width="3000" height="2250" }

LightGCN과 GCN/[Correct and Smooth](/posts/CS224w-05-Message-Passing-and-Node-Classification/)에는 두 가지 차이점이 존재합니다. 먼저 LightGCN에는 Self-loop가 포함되지 않습니다. Multi-scale diffusion단계에서 self-loop가 포함될 뿐입니다. 그다음, final embedding은 모든 layer embedding의 평균이라는 점에서 차이가 있습니다.

![](/assets/img/posts/80/13.webp){: width="3000" height="2250" }

그다음으로 LightGCN이 MF와 다른 점은 LightGCN에서는 user/item embedding이 여러 층에 걸쳐 diffuse된다는 것입니다. Shallow encoder만 학습된다는 점은 같습니다.

![](/assets/img/posts/80/14.webp){: width="3000" height="2250" }

LightGCN의 요약입니다. GNN의 parameter를 제거함으로 NGCF에서 단순화를 진행했습니다.

## PinSAGE

![](/assets/img/posts/80/15.webp){: width="3000" height="2250" }

PinSAGE의 key idea는 nearby node에서 information을 borrow하는 것입니다. 대규모 graph에서 학습하기 위해 mini-batch 내에서 shared negative samples를 사용했습니다. 또한, hard negative sample을 도입했으며, curriculum learning을 활용했습니다. 정리하면 negative sampling technique을 shared negative samples을 통해서 발전시켰습니다.

![](/assets/img/posts/80/16.webp){: width="3000" height="2250" }

BPR loss의 경우 user마다 하나의 positive item과 negative item set을 샘플링해야 합니다. 이때 많은 negative를 샘플링할수록 성능은 좋아지나 expensive해지는 단점이 있습니다. 이런 문제는 mini-batch 내에서 negative sample을 공유하는 방식으로 해결했습니다.

![](/assets/img/posts/80/17.webp){: width="3000" height="2250" }

위 슬라이드 내용처럼 mini-batch 내에서 모두 같은 negative sample을 share하도록 합니다. 이때 성능은 비슷하게 유지됩니다.

![](/assets/img/posts/80/18.webp){: width="3000" height="2250" }

![](/assets/img/posts/80/19.webp){: width="3000" height="2250" }

Curriculum learning에서는 학습이 진행될수록 더 어려운 학습 데이터 셋을 제공합니다. 학습이 진행될수록 # hard negative sample을 더 추가하여 어려운 학습 데이터 셋을 생성합니다.

![](/assets/img/posts/80/20.webp){: width="3000" height="2250" }

보통 item의 개수가 추천하는 item 개수에 비해 많기 때문에 negative를 샘플링해도 easy negative가 대부분인 문제가 있습니다. 모델이 fine-grained 되기 위해서는 hard negative가 필요한 상황입니다.

![](/assets/img/posts/80/21.webp){: width="3000" height="2250" }

User에게 hard negative란 user와 그래프 상에서 가까우면서도 연결되어 있지는 않은 노드입니다. User의 hard negative를 구할 때는 personalized page rank를 계산하고, PPR score가 높은 샘플 중에서 랜덤으로 고르게 됩니다. (너무 높지는 않은 node를 고릅니다.)

![](/assets/img/posts/80/22.webp){: width="3000" height="2250" }

정리하자면, PinSAGE는 rich node attribute과 graph structure를 활용해 high-quality embedding을 생성했을 뿐 아니라 negative sampling strategies를 통해 학습을 효율적으로 진행했습니다.
