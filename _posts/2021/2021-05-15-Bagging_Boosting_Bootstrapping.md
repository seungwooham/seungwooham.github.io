---
title: "Bagging, Boosting, Bootstrapping의 차이"
date: 2021-05-15 00:00:00 +0900
categories: [ML/DL, Ensemble]
tags: [Korean, bagging, boosting, bootstrapping, ensemble-learning]
---
## Bagging, Boosting, Bootstrapping

최근에 친구들과 대화를 나누다가 우연히 Boosting이라는 단어가 대화에 등장하였습니다. 이름은 많이 들어봤는데 정확히 뭔지는 모르는 알고리즘이어서, 이번 기회에 다루어 보려 합니다. 유사한 이름으르 갖고 있어 헷갈리는 Bootstraping, Bagging도 함께 다루어 봅니다.

### 앙상블 학습 (Ensemble Learning)

앙상블 학습(Ensemble learning)은 여러 개의 분류기(Classifier)를 생성하고 각 분류기의 예측 결과를 결합하여 단일 분류기에 비해 정확한 예측을 도출하는 기법을 말합니다. 앙상블 학습에는 크게 세 가지 유형이 있습니다.

- Voting
- Bagging
- Boosting

의사결정나무(Decision Tree Analysis) 알고리즘 중에서 랜덤 포레스트(Random Forest)는 bagging, 나머지 Gradient Boosting, Light GBM(Gradient Boosting Machine), XGBoost는 boosting을 이용한 방법입니다.

### Bootstrapping

복원 추출 방법으로, 전체 데이터의 일부(보통 2/3)만을 추출하고 나머지는 out-of-bag instance라고 칭하며 사용하지 않는 방법입니다. Train 과정에서 out-of-bag instance가 제외되기 때문에 따로 cross validation할 필요 없이 out-of-bag instance로 evaluation을 할 수 있습니다. Bootstrap은 통계에서 표본 분포를 구하기 위해 데이터를 여러 번 복원 추출 하는 것을 의미합니다.

### Bagging

Bootstrap aggregating이라고도 불리며, variance를 줄여 과적합을 방지합니다. Bootstrapping으로 얻어진 모델을 여러 개 활용하여 equal weight vote를 시행하면 ensemble의 효과와 함께 좋은 모델을 만들 수 있게 됩니다.

### Boosting

Boosting은 ensemble 방법으로 weak learner 여러 개를 합쳐 strong learner를 만드는 방법입니다. Bias와 variance를 줄이는 것에 사용됩니다. Weighted majority voting(분류 문제)나 weighted sum(회귀 문제)를 통해 해당 작업을 진행하며 Ada boost와 Gradient boosting이 대표적인 방법이라고 합니다.

### 출처
- https://www.quora.com/What-is-the-difference-between-boost-ensemble-bootstrap-and-bagging
- https://assaeunji.github.io/ml/2020-08-06-tree/
