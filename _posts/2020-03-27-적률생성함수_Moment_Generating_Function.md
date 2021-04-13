---
title: "적률생성함수 (Moment Generating Function)"
date: 2020-03-27
categories: 
  - Machine Learning
tags:
  - 한글
---
### 적률생성함수 (Moment Generating Function)

적률생성함수에서 '적률'은 한자로는 '積率'으로 '쌓을 적'과 '비율 률'자를 써서 만들어진 단어입니다. 적률은 '확률 분포의 위치나 모양 따위의 특성을 나타내는 기댓값'으로 확률 분포에 대한 설명이 담겨있는 값 정도로 풀어서 이야기 할 수 있을 것 같습니다. 김우철 교수님의 수리통계학에서는 적률을 moment라고 설명하고 있습니다. 다른 분야에서도 볼 수 있는 moment generation function과 동일시 할 수 있는 개념이 적률 생성 함수라고 생각하시면 될 것 같습니다.

적률생성함수는

$$M(t)=\sum^{\infty}_{x=0}e^{tx}f(x)\qquad (x=0, 1, 2,\;\cdots)$$

의 식으로 정의 되는데, 위의 식을 계속 미분해가면 아래와 같습니다.

$$\begin{align*}
&\bigg[\frac{d}{dt}M(t)\bigg]_{t=0}=\sum^{\infty}_{x=0}xf(x) \\
&\bigg[\frac{d^2}{dt^2}M(t)\bigg]_{t=0}=\sum^{\infty}_{x=0}x^2f(x) \\
&\qquad\qquad\qquad\vdots\\
&\bigg[\frac{d^k}{dt^k}M(t)\bigg]_{t=0}=\sum^{\infty}_{x=0}x^kf(x) \\
&\qquad\qquad\qquad\vdots\\
\end{align*}$$

가 성립합니다.

이때 $$\sum^{\infty}_{x=0}x^kf(x)$$를 $$f$$의 $$k$$차 적률(moment)이라 부르고, 이는 $$f$$가 확률변수 $$X$$의 분포를 나타낼 때,

$$E(X^{k})\qquad (k=0, 1, 2,\;\cdots)$$

와 같습니다. 이러한 뜻에서 $$M(t)$$를 $$f$$의 **적률생성함수** (moment generatinf function)라고 부릅니다.

### 출처
김우철. 수리통계학 = Mathematical Statistics / 김우철 지음, 2012.

<!-- <script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> -->