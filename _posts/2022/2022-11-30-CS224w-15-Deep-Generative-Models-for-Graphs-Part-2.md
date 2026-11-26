---
title: "CS224w - 15. Deep Generative Models for Graphs Part 2"
date: 2022-11-30 19:07:31 +0900
categories: [ML/DL, CS224W]
tags: [Korean, CS224W, GNN, graph-generation]
original_url: https://seungwooham.tistory.com/84
---

## Scaling Up Graph Generation

![](/assets/img/posts/84/01.webp){: width="3000" height="2250" }

지금의 GraphRNN은 어떤 node가 이전에 존재하던 node들과 연결될 수 있을지를 모두 계산해야 합니다. Computation time상 intractable한데, BFS node ordering으로 이를 일부 해결할 수 있습니다.

![](/assets/img/posts/84/02.webp){: width="3000" height="2250" }

![](/assets/img/posts/84/03.webp){: width="3000" height="2250" }

간단하게 전의 M개의 node만 보겠다는 의미인데, 걱정이 되기는 합니다. M개보다 더 이전의 node와 연결해야 하는 경우를 놓칠 것 같지만, BFS ordering에서는 새 node가 BFS frontier(queue)에 있는 node와만 연결될 수 있으므로 M을 BFS queue의 최대 크기로 잡으면 놓치는 edge가 없습니다.

## Evaluating Graph Generation

![](/assets/img/posts/84/04.webp){: width="3000" height="2250" }

Graph의 집합(set)을 비교하는 방법을 생각해봅시다. 여기서는 graph의 similarity metric을 정의해야 합니다. Visual similarity일수도 있고, graph statistic이 비슷할 수도 있습니다.

![](/assets/img/posts/84/05.webp){: width="3000" height="2250" }

![](/assets/img/posts/84/06.webp){: width="3000" height="2250" }

정성적인 visual similarity에서 GraphRNN이 다른 Baseline에 비해 잘 작동합니다.

![](/assets/img/posts/84/07.webp){: width="3000" height="2250" }

보다 정량적인 graph statistics similarity를 확인해봅시다. 직접 두 graph를 비교하는 isomorphism test는 NP 문제입니다. 따라서 Graph statistics를 비교합니다. Degree distribution, clustering coefficient distribution, orbit count statistics 등은 모두 일종의 distribution 입니다.

![](/assets/img/posts/84/08.webp){: width="3000" height="2250" }

첫 step에서 두 graph statistic을 비교하는 방법으로 Earth Mover Distance (EMD)를 제시합니다. 각 statistic을 토양으로 보고 토양을 얼마나 옮겨야 두 statistic이 같아지는지 측정합니다.

![](/assets/img/posts/84/09.webp){: width="3000" height="2250" }

두 번째로 graph statistic의 set을 비교합니다. 이때 MMD라는 distance measure를 활용합니다. 각 feature의 mean embedding 사이의 거리를 파악한다고 보시면 되겠습니다.

## Application of Deep Graph Generative Models to Molecule Generation

![](/assets/img/posts/84/10.webp){: width="3000" height="2250" }

말이 되고, 실제로 존재할만한 molecule이 어떤 property score를 만족하도록 만들 수 있을까요? 먼저 특정 objective에 대해 optimize시켜서 어떤 성질을 이끌어낼 수 있습니다. 그다음 rule로 valid 여부를 확인합니다. 실제 example을 통해서 학습하면 realistic한 데이터를 얻을 수 있습니다.

![](/assets/img/posts/84/11.webp){: width="3000" height="2250" }

이 작업을 위해서 강화 학습이 등장합니다. Reinforcement learning은 ML agent가 environment를 관찰하고, 어떤 action을 취해 reward를 얻는 것입니다. 이를 graph generation에 적용합니다.

![](/assets/img/posts/84/12.webp){: width="3000" height="2250" }

그렇게 나온 연구가 GCPN: Graph Convolutional Policy Network입니다. GCPN의 주요 component는 다음과 같습니다.

- GNN: Graph의 structural information을 capture
- RL: 원하는 objective를 얻도록 generation guide를 제공
- Supervised training: 주어진 dataset의 example을 imitate하도록 유도

![](/assets/img/posts/84/13.webp){: width="3000" height="2250" }

GCPN과 GraphRNN은 graph를 순차적으로 generate한다는 점과 주어진 graph dataset을 imitate한다는 점에서 공통점을 갖습니다. 하지만 GCPN은 RNN이 아닌 GNN을 사용해 generation action을 예측합니다. 또한 GCPN은 RL을 이용하여 graph generation을 guide합니다.

![](/assets/img/posts/84/14.webp){: width="3000" height="2250" }

위는 GCPN의 전체적인 과정입니다. Reward 설정할 때는 두 가지를 활용합니다.

- Step reward: Valid action을 선택하도록, small reward를 설정
- Final reward: Desired property를 optimize 하도록 big reward를 설정

![](/assets/img/posts/84/15.webp){: width="3000" height="2250" }

Supervised training 단계에서는, real observed graph에서 얻은 action을 imitate하도록 policy를 학습합니다. 이때 gradient-based training이 진행됩니다.

RL training 단계에서는 reward optimization을 기반으로 한 policy gradient-based training이 진행됩니다.
