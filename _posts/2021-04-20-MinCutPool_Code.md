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

```python
(batch_size, num_nodes, _), k = x.size(), s.size(-1)

s = torch.softmax(s, dim=-1)

if mask is not None:
    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask
```
x.size()는 [1, 500, 2]이니 batch_size로 1을 받고, num_nodes로 500을 받습니다. Feature의 차원인 2는 버립니다. s.size()는 [1, 500, 5]이니 -1 index에 해당하는 5를 k에 받아옵니다. s는 마지막 차원에 대해서 softmax 연산을 진행합니다. [1, 0, 0, 0, 0]이었던 값은 [0.4046, 0.1488, 0.1488, 0.1488, 0.1488]으로 변화합니다. mask는 0과 1로 이루어진 [B, N]차원 행렬로, 사용할 node를 고릅니다. 이번에는 사용하지 않도록 하겠습니다.

```python
out = torch.matmul(s.transpose(1, 2), x)
out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
```
이론편에서 봤던 transpose 부분입니다. Batch에 해당하는 0번째 차원은 두고, 1번째, 2번째 차원을 transpose하고 x와 행렬곱을 시행합니다. [1, 5, 500]과 [1, 500, 2]가 곱해져 [1, 5, 2]가 됩니다. out_adj도 이론편에서 봤던 식을그대로 시행합니다. s의 transpose와 adj를 행렬곱하고, 이를 다시 행렬곱합니다. [1, 5, 500] * [1, 500, 500] * [1, 500, 5]의 곱이 시행되어 결과는 [1, 5, 5]가 됩니다. $$\mathbf{X}^{pool}$$과 $$\mathbf{A}^{pool}$$를 구했으니 잠시 다른 함수를 정의해봅시다.

```python
def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out
```
Einsum을 이용한 함수입니다. 아직 einsum이 익숙하시지 않은 분은 [이 글](<https://seungwooham.github.io/machine%20learning/Einsum_%EC%9E%85%EB%AC%B8%ED%95%98%EA%B8%B0_Introduction_to_Einsum/>)을 확인해보세요.


### 출처
- Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>
- https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/mincut_pool.html#dense_mincut_pool