---
title: "포아송분포의 적률생성함수 (Moment Generating Function of Poisson Distribution)"
date: 2020-03-27
categories:
  - Machine Learning
tags:
  - 한글
---
### 포아송분포의 적률생성함수 (Moment Generating Function of Poisson Distribution)

포아송분포 $$Poisson(\lambda)$$의 적률생성함수에 대하여 알아봅시다.

적률생성함수에 대해서 아직 모르시는 분들은 [여기]를 확인해보시면 됩니다.

[여기]: https://seungwooham.github.io/%ED%86%B5%EA%B3%84/%EC%A0%81%EB%A5%A0%EC%83%9D%EC%84%B1%ED%95%A8%EC%88%98_Moment_Generating_Function/

$$e^{a}=\sum^{\infty}_{x=0}\frac{a^x}{x!}$$

인 사실을 이용하면 다음과 같이 구할 수 있습니다.

$$\begin{align*}
M(t)&=\sum^{\infty}_{x=0}e^{tx}f(x)=\sum^{\infty}_{x=0}e^{tx}\frac{\lambda^{x}e^{-\lambda}}{x!} \\
&=e^{-\lambda}\sum^{\infty}_{x=0}\frac{(\lambda e^t)^x}{x!} \\
&=e^{-\lambda}e^{\lambda e^t} \\
&=e^{\lambda (e^t-1)}
\end{align*}$$

---
### 포아송분포 $$\mathbf{Poisson(\lambda)}$$의 적률생성함수
$$M(t)=e^{\lambda (e^t-1)}$$

---

포아송분포 $$Poisson(\lambda)$$의 적률생성함수를 이용하면 평균과 분산을 쉽게 구할 수 있습니다. 적률생성함수의 미분값을 확인하여봅시다.

$$\begin{align*}
&\frac{d}{dt}M(t)=\lambda e^{t} e^{\lambda (e^t-1)} \\
&\frac{d^2}{dt^2}M(t)=\lambda e^{t} e^{\lambda (e^t-1)}+(\lambda e^{t})^2 e^{\lambda (e^t-1)}\\
\end{align*}$$

임을 이용하여 $$E(X)$$와 $$E(X^2)$$을 구할 수 있습니다.

$$\begin{align*}
&E(X)=\bigg[\frac{d}{dt}M(t)\bigg]_{t=0}=\lambda\\
&E(X^2)=\bigg[\frac{d^2}{dt^2}M(t)\bigg]_{t=0}=\lambda+\lambda^2\\
\end{align*}$$

이다. 따라서

$$\begin{align*}
Var(X)&=E(X^2)-{E(x)}^2\\
&=\lambda + \lambda^2 - \lambda^2 \\
&=\lambda \\
\end{align*}$$

---
### 포아송분포 $$\mathbf{Poisson(\lambda)}$$의 평균과 분산
$$X \sim Poisson(\lambda)$$일 때

$$E(X)=\lambda, Var(X)=\lambda$$

---

포아송분포는 이항분포의 근사로서 이용될 뿐 아니라, 일정한 시간이나 일정한 공간에서 희귀하게 일어나는 사건의 횟수 등에 관한 확률모형으로 많이 이용됩니다. 예를 들면, 어느 지역의 1일 보행사고 발생 수, 3개월 동안 열차가 탈선할 횟수 등에 포아송분포를 적용할 수 있습니다.

일반적으로 단위당 발생률이 $$\lambda$$인 희귀현상에 대하여, 0에서 $$t$$ 사이의 발생 횟수는 확률분포로서 포아송분포 $$Poisson(\lambda t)$$가 흔히 사용됩니다. 여기서 '희귀'라는 말은 짧은 기간 동안에는 발생 가능성이 작으며, 두 번 이상 발생할 확률이 한 번 발생할 확률에 비하여 매우 작음을 의미합니다.

### 출처
김우철. 수리통계학 = Mathematical Statistics / 김우철 지음, 2012.

<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">