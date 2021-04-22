---
title: "MinCutPool: 코드 분석 (MinCutPool: Understanding the Code)"
date: 2021-04-22
categories:
  - Machine Learning
tags:
  - 한글
---
### MinCutPool: 코드 분석 (MinCutPool: Understanding the Code)

이론편1은 [여기](<https://seungwooham.github.io/machine%20learning/MinCutPool1/>), 이론편2는 [여기](<https://seungwooham.github.io/machine%20learning/MinCutPool2/>)에서 확인하실 수 있습니다.

이번 게시물에서는 MinCutPool의 code를 예시 데이터와 함께 분석해보겠습니다. 제가 사용한 데이터는 마이크로 모빌리티의 이동패턴 데이터인데, 실제 영업에서 사용된 데이터이므로 자세한 공개가 어려워 데이터 차원의 변화를 위주로 설명드리겠습니다. 전체 코드를 보기 전에 제가 정의한 x(feature matrix), adj(adjacecny matrix,), s(cluster assignment matrix)의 차원은 각각 다음과 같습니다.

```python
print(x.shape) # torch.Size([1, 500, 2])
print(adj.shape) # torch.Size([1, 500, 500])
print(s.shape) # torch.Size([1, 500, 5])
```

총 500개의 node로 이루어진 graph이며, x에는 약 1달 동안의 node별 마이크로 모빌리티 대여 및 반납 횟수, adj는 node간 이동 횟수, s는 5차원 one-hot 벡터로 이루어져 있습니다. s의 차원은 임의로 정했습니다. 단계별로 과정을 밟아봅시다.

```python
import torch

EPS = 1e-15

x = x.unsqueeze(0) if x.dim() == 2 else x
adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
s = s.unsqueeze(0) if s.dim() == 2 else s
```

Batch 없이 2차원으로 벡터가 들어올 경우 차원 하나를 추가해줍니다. [500, 2]일 때 [1, 500, 2]로 바꿔주는 것인데 제 input은 모두 3차원이어서 여기서는 딱히 변화가 없습니다.

### 출처
- Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>
- https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/mincut_pool.html#dense_mincut_pool