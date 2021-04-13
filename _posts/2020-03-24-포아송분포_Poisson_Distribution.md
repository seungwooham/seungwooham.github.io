---
title: "포아송분포 (Poisson Distribution)"
date: 2020-03-24
categories:
  - Machine Learning
tags:
  - 한글
header:
  teaser: "/assets/images/2020-03-24-PD1.png"
---
### 포아송분포 (Poisson Distribution)

이항분포가 적용될 수 있는 실제 문제에서 $$n$$이 충분히 크고 성공률 $$p$$는 충분히 작은 때가 많이 있습니다. 예를 들어 자동차 사고에 의한 사망자수, 보험회사의 보험금 지급 건수 등이 그러한 경우입니다. 이러한 경우에 이항 분포의 확률을 정확히 계산하는 것은 매우 어려운 일이 됩니다.

이항확률의 근사계산에 대하여 알아봅시다. 표현의 간결성을 위하여 $$np=\lambda$$라는 일정한 값을 유지하면서 $$n$$이 충분히 크고 $$p$$가 충분히 작다고 합시다.

수학에서 알려진 사실인

$$\lim_{n\rightarrow\infty} (1+\frac{a}{n})^{n} = e^{a}$$

를 이용하면, 이항확률은

$$\begin{align*}
\begin{pmatrix}n\\x\end{pmatrix}p^x(1-p)^{n-x}&=n(n-1)\cdots(n-x+1)p^x(1-p)^{n-x}\cdot\frac{1}{x!} \\
&=\frac{n}{n}(1-\frac{1}{n})\cdots(1-\frac{x-1}{n})(np)^x(1-\frac{\lambda}{n})^n\cdot(1-\frac{\lambda}{n})^{-x}\cdot\frac{1}{x!} \\
&\fallingdotseq\lambda^{x}e^{-\lambda}\frac{1}{x!}
\end{align*}$$

와 같이 근사시킬 수 있습니다. 이와 같은 이항확률의 근사를 포아송 (Poisson) 근사라고 합니다.

---
### 이항확률의 포아송 근사
$$n$$이 충분히 크고 $$p$$가 충분히 작으면서 $$np=\lambda$$이면
$$\begin{pmatrix}n\\x\end{pmatrix}p^{x}(1-p)^{n-x}\fallingdotseq\frac{\lambda^{x}}{x!}\cdot e^{-\lambda}$$

---

한편 지수함수의 테일러 급수 전개에 의하면

$$e^{\lambda}=\sum^{\infty}_{x=0}\frac{\lambda^{x}}{x!}$$

이며, 이로부터

$$f(x)=\frac{\lambda^{x}}{x!}e^{-\lambda} \qquad x=0, 1, 2,\cdots$$

로 정의되는 함수 $$f(x)$$는 항상 양수이고, 그 전체 합이 1이라는 확률질량 함수의 성질을 만족하는 것을 알 수 있습니다.

이와 같은 확률질량함수를 갖는 분포를 포아송분포 (Poisson distribution)라고 부르며, 기호로는 $$X\sim Poisson(\lambda)$$로 나타냅니다.

---
### 포아송분포 $$\mathbf{Poisson(\lambda)}$$의 확률질량함수
$$f(x)=\frac{\lambda^x}{x!}e^{-\lambda} \qquad x=0, 1, 2, \cdots$$

---

포아송분포의 형태는 $$\lambda$$까지 확률이 증가하고, 그 이후에는 감소하는 모습을 하고 있습니다.

![Figure_1](/assets/images/2020-03-24-PD1.png)

### 출처
김우철. 수리통계학 = Mathematical Statistics / 김우철 지음, 2012.