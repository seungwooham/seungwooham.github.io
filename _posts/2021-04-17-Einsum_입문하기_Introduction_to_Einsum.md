---
title: "Einsum 입문하기 (Introduction to Einsum)"
date: 2021-04-17
categories:
  - Machine Learning
tags:
  - 한글
---
### Einsum 입문하기 (Introduction to Einsum)

오늘은 pytorch 코드를 작성하다보면 종종 마주치는 einsum에 대해 간략히 소개해보려합니다. Einsum은 아인슈타인 표기법(Einstein summation convention)을 이용하여 연산을 수행하는 함수입니다. Pytorch외에 numpy와 tensorflow 코드에서도 유사한 방법으로 사용되어서 알아두면 다른 프레임워크를 사용하실 때에도 두고두고 도움이 됩니다.

우리는 반복적인 연산을 $$\Sigma$$ 표기법을 이용하여 간략히 나타냅니다. Einsum 표기법은 $$\Sigma$$를 이용한 연산 표기와 같은 형태를 갖습니다. 다만 $$\Sigma$$ 기호를 생략한다는 차이가 있습니다.

Einsum은 다양한 종류의 연산에 대해 동일한 형태로 표현할 수 있다는 장점이 있습니다. 마치 우리가 $$\Sigma$$를 이용하여 다양한 연산을 표기하는 것과 같습니다. 게다가 연산 속도도 빠릅니다. (2020년 말의 업데이트를 통해서 pytorch, numpy 안의 built-in function 보다도 빠른 속도를 제공합니다. [이와 관련된 pytorch issue를 확인할 수 있습니다.](<https://github.com/pytorch/pytorch/issues/32591/>)) 하지만 처음 einsum을 접하면 불친절할 정도로 간략한 표기에 어려움을 느끼게 됩니다.

가장 간단한 예시를 들어보겠습니다. 우리가 행렬 A와 행렬 B를 곱할 때 $$\Sigma$$를 이용하여 표기하는데, 사실 $$\Sigma$$ 없이도 상식적으로 어떻게 연산이 진행되는지는 알수 있기 때문에 다음과 같이 나타낼 수 있습니다.

$$M_{ij}=\sum_{k}A_{ik}B_{kj}:= A_{ik}B_{kj}$$

이를 코드로 작성하면 아래와 같습니다.

```python
import numpy as np

A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.empty((3, 2))

for i in range(A.shape[0]):
  for j in range(B.shape[1]):
    total = 0
    for k in range(A.shape[1]):
      total += A[i,k]*B[k,j]
    
    M[i,j] = total
```

별 내용이 없는데 코드는 깁니다. 같은 코드를 einsum 표기를 활용하면 아래와 같이 됩니다.

```python
import numpy as np

A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.einsum('ik,kj->ij', A, B)
```

위 예시에서 A의 index인 i와 k, 그리고 B의 index인 k와 j 중에서 output matrix M에 남아있는 index는 i와 j입니다. k는 연산 과정에서 사라집니다. i와 j처럼 output에 남아있는 index를 free index라 하고, k처럼 연산 과정에서 사라지는 index를 summation index라고 합니다. for 문을 활용한 code를 보면 free index가 바깥쪽 for문에, summation index가 가장 안쪽 for문에 있음을 알 수 있습니다. Einsum 표현이 헷갈릴 때 for문으로 구성하면 어떻게 될지를 예상해보면 쉽게 정리 될 때가 많다고 합니다.

Einsum의 규칙으로 크게 네 가지를 제시할 수 있습니다.

1. 화살표 왼쪽에는 연산 대상의 index, 화살 오른쪽에는 연산 결과물의 index가 옵니다. 작은 따옴표(') 구문이 끝난 뒤에는 연산 대상을 지시합니다.
  > ex) M = np.einsum('ik,kj->ij', A, B)
    - 연산 대상의 index: ik, kj
    - 연산 결과물의 index: ij
    - 연산 대상: A, B
2. Einsum 표기에서 반복되는 index는 free index로 해당 축에서 행렬 연산이 이루어짐을 의미합니다.
  > ex) M = np.einsum('ik,kj->ij', A, B)
    - free index: k
3. Index가 생략되는 경우, 이 때는 해당 index를 기준으로 합이 이루어짐을 의미합니다. 
  > ex) sum = np.einsum('i->', x)
    - summation index(axis): i
4. 연산이 이루어지지 않는 축은 어떤 순서로 transpose해도 무방합니다.
  > ex) reorder_axis = np.einsum('ijk->kji', x)

그럼 이제 예시들을 보며 마무리 해봅시다.
```python
import torch

x = torch.rand((2, 3))
y = torch.rand((2, 3, 5))
v = torch.rand((1, 3))

# 전치 행렬 (3차원 이상의 경우 축 순서 변환)
# Permutation ot tensors
print(torch.einsum("ij->ji", x))
print(torch.einsum("ijk->kij", y))

# 행렬의 합
# Summation of matrix
print(torch.einsum("ij->", x))
print(torch.einsum("ijk->", y))

# 열의 합
# Summation of column
print(torch.einsum("ij->j", x))

# 행의 합
# Summation of row
print(torch.einsum("ij->i", x))

# 행렬과 벡터의 곱 (행렬 곱)
# Matirx-Vector multiplication
print(torch.einsum("ij,kj->ik", x, v))

# 행렬과 행렬의 곱 (행렬 곱)
# Matirx-Matrix multiplication
print(torch.einsum("ij,kj->ik", x, x))
# 자연스럽게 transpose를 한 점이 인상적입니다.

# 행렬 내부 특정 행의 내적
# Dot product first row with first row of matrix
print(torch.einsum("i,i->", x[0], x[0]))
print(torch.einsum("i,i->", y[0][0], y[0][0]))

# 행렬과 행렬의 내적
# Dot product of matrix
print(torch.einsum("ij,ij->", x, x))
print(torch.einsum("ijk,ijk->", y, y))

# 원소 단위의 곱
# Hadamard product (element-wise multiplication of matrix)
print(torch.einsum("ij,ij->ij", x, x))
print(torch.einsum("ijk,ijk->ijk", y, y))

# 외적
# Outer product
a = torch.rand((3))
b = torch.rand((5))
print(torch.einsum("i,j->ij", a, b))

# 배치 행렬 곱
# Batch matrix multiplication
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
print(torch.einsum("ijk,ikl->ijl", a, b))
# i는 그대로 유지한 상태에서 자연스럽게 [2*5]*[5*3]의 행렬 곱을 3회 시행했습니다.

# 행렬의 대각성분
# Matrix diagonal
a = torch.rand((3, 3))
print(torch.einsum("ii->i", a))

# 행렬 대각성분의 합
# Matrix trace
print(torch.einsum("ii->", a))
```

### 출처
https://www.youtube.com/watch?v=pkVwUVEHmfI