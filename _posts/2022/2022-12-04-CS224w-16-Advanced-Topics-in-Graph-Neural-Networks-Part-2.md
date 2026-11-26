---
title: "CS224w - 16. Advanced Topics in Graph Neural Networks Part 2"
date: 2022-12-04 18:39:25 +0900
categories: [ML/DL, CS224W]
tags: [Korean, CS224W, GNN, expressiveness]
original_url: https://seungwooham.tistory.com/86
---

## Identity-Aware Graph Neural Networks

![](/assets/img/posts/86/01.webp){: width="3000" height="2250" }

![](/assets/img/posts/86/02.webp){: width="3000" height="2250" }

![](/assets/img/posts/86/03.webp){: width="3000" height="2250" }

서로 다른 node이지만 같은 computational graph를 가져서 같은 embedding으로 계산되는 문제가 다양한 graph에서 나타납니다. Node, edge, graph level의 task에서 모두 보이는 문제점들입니다.

![](/assets/img/posts/86/04.webp){: width="3000" height="2250" }

이에 대한 해결방법으로 root node에 대해서 colorize한 뒤, computational graph에 colorize를 반영하는 방법이 있습니다. 이 경우 Graph 구조가 다르다면, computational graph에서 colorize 되는 부분이 달라지기 때문에 구별이 가능합니다.

![](/assets/img/posts/86/05.webp){: width="3000" height="2250" }

![](/assets/img/posts/86/06.webp){: width="3000" height="2250" }

Edge level과 graph level에 대해서도 똑같이 적용할 수 있습니다.

![](/assets/img/posts/86/07.webp){: width="3000" height="2250" }

다른 종류(color)의 node에 대해서 다른 network로 aggregate 한다면 computation graph가 같아도 다른 embedding을 만들어낼 수 있습니다. 따라서 node color에 따라 서로 다른 neural net A, B를 적용하여 embedding을 구분할 수 있습니다.

![](/assets/img/posts/86/08.webp){: width="3000" height="2250" }

ID-GNN을 사용하면 cycle 수도 구할 수 있습니다. Colorize된 root node가 몇 번째 level에서 몇 개 등장하는지 counting하면 cycle 수가 자연스럽게 구해집니다.

![](/assets/img/posts/86/09.webp){: width="3000" height="2250" }

각 hop에 포함된 colorize된 node의 수 자체를 node feature로 활용하는 방법도 존재합니다. 이걸 Leskovec 교수님 연구실에서 제안한 것으로 보입니다.

![](/assets/img/posts/86/10.webp){: width="3000" height="2250" }

이번 장의 요약 내용입니다.

## Robustness of Graph Neural Networks

과연 GNN은 attack에 robust할까요? 이번 장을 통해서 확인하여봅시다. Recommender system, social network등에서 attack을 가하며 robustness를 알아보겠습니다.

![](/assets/img/posts/86/11.webp){: width="3000" height="2250" }

Attack의 종류에는 attacker가 target node를 직접 조작할 수 있는 direct attack과 간접적으로 node, edge등을 조작하는 indirect attack이 있습니다.

![](/assets/img/posts/86/12.webp){: width="3000" height="2250" }

위는 direct attack에 대한 예시로, target의 feature를 바꾸거나, edge를 추가/제거하는 모습을 보여줍니다.

![](/assets/img/posts/86/13.webp){: width="3000" height="2250" }

위 그림은 indirect attack으로, neighbor의 node feature, edge를 수정하게 됩니다.

![](/assets/img/posts/86/14.webp){: width="3000" height="2250" }

Adversarial attack을 할 수 있도록 graph를 바꾸려면 어떻게 해야 할까요? 가장 높은 확률로 예측되는 class에 대한 확률을 줄이고, 다른 class에 대한 확률을 높이면 됩니다.

![](/assets/img/posts/86/15.webp){: width="3000" height="2250" }

원래의 graph에서 adjacency matrix나 feature matrix에 조금 조작을 가합니다. 기존의 graph statistic은 최대한 유지하는 것이 목적입니다.

![](/assets/img/posts/86/16.webp){: width="3000" height="2250" }

![](/assets/img/posts/86/17.webp){: width="3000" height="2250" }

Target node에 대한 class prediction이 graph manipulation 이후 달라지도록 유도합니다.

![](/assets/img/posts/86/18.webp){: width="3000" height="2250" }

새로운 class로 분류될 확률을 증가시키고, 원래 class로 분류될 확률을 감소시키도록 학습합니다. 위의 델타로 정의된 값이 가장 커지면 됩니다.

![](/assets/img/posts/86/19.webp){: width="3000" height="2250" }

인접 행렬의 경우 discrete하여 gradient-based learning이 적용되기 힘든데, 이를 근사한 연구도 존재합니다.

![](/assets/img/posts/86/20.webp){: width="440" height="330" }
![](/assets/img/posts/86/21.webp){: width="440" height="330" }

위 그림에서 알 수 있다시피, 2800개의 node, 8000개의 edge가 있는 graph에서 target node와 붙어있는 5개의 edge를 바꿨더니 예측하는 class가 완전히 달라졌습니다.

![](/assets/img/posts/86/22.webp){: width="3000" height="2250" }

Direct attack이 가장 효과적인 결과를 내는 것을 알 수 있습니다. Figure에서 classification margin이 작을수록 잘못된 class의 prediction probability가 상대적으로 큰 것입니다.

![](/assets/img/posts/86/23.webp){: width="3000" height="2250" }

이번 장의 요약입니다.
