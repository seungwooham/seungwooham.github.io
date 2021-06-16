---
title: "MinCutPool: 코드 분석 (MinCutPool: Understanding the Code)"
date: 2021-04-22
categories:
  - Machine Learning
tags:
  - 한글
---
## MinCutPool: 코드 분석 (MinCutPool: Understanding the Code)

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
Einsum을 이용한 함수입니다. 아직 einsum이 익숙하시지 않은 분은 [이 글](<https://seungwooham.github.io/machine%20learning/Einsum_%EC%9E%85%EB%AC%B8%ED%95%98%EA%B8%B0_Introduction_to_Einsum/>)을 확인해보세요. _rank3_trace(x)는 행렬의 trace를 구해줍니다. ijj에서 첫 i는 batch 구분을 위해서 존재합니다. 그 뒤로는 행렬의 $$a_{jj}$$를 더합니다. 제 예시에서는 batch가 1차원이어서 x는 [1, 1]차원의 scalar같은 행렬이 됩니다.
eye는 일단 node 개수와 같은 identity matrix입니다. _rank3_diag는 x의 원소가 diagonal인 matrix를 생성합니다.

```python
# MinCUT regularization.
mincut_num = _rank3_trace(out_adj)
d_flat = torch.einsum('ijk->ij', adj)
d = _rank3_diag(d_flat)
mincut_den = _rank3_trace(
    torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
mincut_loss = -(mincut_num / mincut_den)
mincut_loss = torch.mean(mincut_loss)
```
이제 cut loss에 대한 부분입니다. 새롭게 얻어진 out_adj($$\mathbf{A}^{pool}$$)의 trace를 구합니다. 그 다음 대각 행렬을 만들기 위해서 adj의 행별 합을 만든 이후, _rank3_diag 함수로 대각 행렬 d를 생성합니다. 이 D를 s의 transpose와 d, 그리고 s와 행렬곱하면 우리가 아는 cut loss의 분모 부분이 생성됩니다. 아래 식과 함께 비교하면서 보시면 이해가 쉽게 되실 겁니다.

$$\begin{aligned}
\mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A} \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D} \mathbf{S})}
\end{aligned}$$

```python
# Orthogonality regularization.
ss = torch.matmul(s.transpose(1, 2), s)
i_s = torch.eye(k).type_as(ss)
ortho_loss = torch.norm(
    ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
    i_s / torch.norm(i_s),
    dim=(-1, -2))
ortho_loss = torch.mean(ortho_loss)
```
이 다음은 orthogonality loss입니다. 마찬가지로 식의 모습을 그대로 따라갑니다. 

$$\begin{aligned}
\mathcal{L}_o = { \left\| \frac{\mathbf{S}^{\top} \mathbf{S}}{ { \| \mathbf{S}^{\top} \mathbf{S} \| }_F } - \frac{\mathbf{I}_C}{\sqrt{C}} \right\| }_F
\end{aligned}$$

저는 예시로 만든 cluster가 orthogonal해서 ortho_loss가 0이 나와 버렸습니다. ([1, 0, 0, 0, 0] 100개, [0, 1, 0, 0, 0] 100개, ... , [0, 0, 0, 0, 1] 100개) 

```python
# Fix and normalize coarsened adjacency matrix.
ind = torch.arange(k, device=out_adj.device)
out_adj[:, ind, ind] = 0
d = torch.einsum('ijk->ij', out_adj)
d = torch.sqrt(d)[:, None] + EPS
out_adj = (out_adj / d) / d.transpose(1, 2)
```

이렇게 말하기 민망하지만 아래의 식과 완전히 동일하게 코드가 짜여있어서 추가 설명이 사족이 될 듯 합니다. 첫 두 줄이 ('out_adj[:, ind, ind] = 0'까지) $$\hat{\mathbf{A}}$$에 대한 부분입니다.

$$\begin{aligned}
& \hat{\mathbf{A}} = \mathbf{A}^{pool}-\mathbf{I}_K \mathrm{diag}(\mathbf{A}^{pool}) \\
& \mathbf{A}^{pool} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}
\end{aligned}$$

지금까지는 이 연산 한 번을 거치면 자동으로 cluster가 된다고 생각하고 있었는데, 공식 문서에 나와있듯이 이건 하나의 layer이고, 이 layer를 여러 번 반복해야지만 적절한 결과를 얻을 수 있습니다. 아래는 전체 코드입니다.

```python
import torch

EPS = 1e-15

def dense_mincut_pool(x, adj, s, mask=None):

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss



def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out
```

return 되는 out은 $$\mathbf{X}^{pool} = {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot \mathbf{X}$$의 $$\mathbf{X}^{pool}$$ 입니다. out_adj는 $$\tilde{\mathbf{A}}^{pool} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2}$$의 $$\tilde{\mathbf{A}}^{pool}$$입니다. mincut_loss와 ortho_loss는 각각 loss를 나타냅니다.

간단한 논문이었는데 자세하게 해석하다보니 세 개의 게시물이 올라갔습니다. 다음에는 attention과 연관된 graph 논문을 다루어보고자 합니다.

### 출처
- Bianchi, Filippo Maria, Daniele Grattarola, and Cesare Alippi. "Spectral clustering with graph neural networks for graph pooling." In International Conference on Machine Learning, pp. 874-883. PMLR, 2020. <br/>
- https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/dense/mincut_pool.html#dense_mincut_pool