---
title: "CS224w - 14. Community Detection in Networks Part 2"
date: 2022-11-21 17:46:49 +0900
categories: [ML/DL, CS224W]
tags: [Korean, CS224W, GNN, community-detection]
original_url: https://seungwooham.tistory.com/82
math: true
---

## Louvain Algorithm

![](/assets/img/posts/82/01.webp){: width="3000" height="2250" }

Louvain algorithm은 community detection에 활용되는 알고리즘입니다. 현재 그래프에서 neighborhood node를 추가하는데, 주위에서 Q를 maximal하게 늘릴 수 있는 node를 greedy하게 search합니다. 해당 node를 찾으면 community에 aggregate합니다.

![](/assets/img/posts/82/02.webp){: width="3000" height="2250" }

Community에 node를 추가하는 과정에서, 기존의 다른 community에 있던 node를 제외시킬수도 있습니다. 따라서 이 과정에서는 D에서 i가 빠져나가는 것과 새로 C로 편입되는 과정의 $$\Delta Q$$를 계산해야 합니다.

![](/assets/img/posts/82/03.webp){: width="3000" height="2250" }

Modularity Q는 식 정리를 통해서 비교적 간단하게 구할 수 있습니다. Community 내부 edge수와 총 edge수를 비교하여 얻을 수 있고, 위의 슬라이드는 그 아이디어에 대한 간략한 정리입니다.

![](/assets/img/posts/82/04.webp){: width="3000" height="2250" }

위와 같은 식으로 Modularity Q를 간단하게 표현할 수 있습니다. 내부의 link와 total link의 비교를 통해서 값이 얻어집니다.

![](/assets/img/posts/82/05.webp){: width="3000" height="2250" }

위 슬라이드 처럼, merge 전후의 Q를 비교하여 Q가 증가하는 방향으로 aggregation을 진행하면 됩니다.

![](/assets/img/posts/82/06.webp){: width="3000" height="2250" }

Phase 1과 Phase 2를 반복적으로 진행하면서 community를 만들게 됩니다.

![](/assets/img/posts/82/07.webp){: width="3000" height="2250" }

이러한 과정으로 modularity를 정의하고, community를 생성할 수 있습니다.

## Detecting Overlapping Communities: AGM & NOCD

![](/assets/img/posts/82/08.webp){: width="3000" height="2250" }

그러면 한 node가 여러 community에 중복해서 들어간 경우의 node community는 어떻게 얻을 수 있을까요?

![](/assets/img/posts/82/09.webp){: width="3000" height="2250" }

우리는 community 소속 정보로 graph 만들고, 그 그래프가 잘 생성되었는지 확인하는 방식으로 여러 community에 포함된 node를 확인할 수 있습니다.

![](/assets/img/posts/82/10.webp){: width="3000" height="2250" }

![](/assets/img/posts/82/11.webp){: width="3000" height="2250" }

Community affiliation을 기반으로 그래프를 생성합니다. 각 community에 속한 node는 서로 연결될 확률이 양수로 특정되고, 이 확률에 의해 여러 community에 함께 속한 node들은 서로 연결됩니다.

![](/assets/img/posts/82/12.webp){: width="3000" height="2250" }

이 과정에서는 community가 주어지면 node 사이의 연결 확률을 만들 수 있는 model이 필요합니다. F가 그 model에 해당합니다.

![](/assets/img/posts/82/13.webp){: width="3000" height="2250" }

우리는 두 node가 community를 통해 연결될 확률을 strength를 의미하는 F를 활용하여 정의할 수 있습니다. Model에서 얻은 embedding으로 strength를 구하는 것입니다.

![](/assets/img/posts/82/14.webp){: width="3000" height="2250" }

여러개의 community에 속해 있어도, dot product로 두 node가 연결될 확률을 비교적 쉽게 계산 가능합니다. 이때 SGD로 update할 수 있습니다.

![](/assets/img/posts/82/15.webp){: width="3000" height="2250" }

이 모델은 올바른 그래프를 생성할 수 있도록 loss를 정의합니다.
