---
title: "Bagging, Boosting, Bootstrapping의 차이"
date: 2021-05-15
categories:
  - Machine Learning
tags:
  - 한글
---
## Bagging, Boosting, Bootstrapping

최근에 친구들과 대화를 나누다가 우연히 Boosting이라는 단어가 대화에 등장하였습니다. 제게는 이름은 많이 들어봤는데 정확히 뭔지는 모르는 알고리즘으로 느껴졌습니다. 그 자리에 있던 친구들 모두 정확한 정의를 몰라서 찾아보려 하는데, B로 시작하는 용어들이 또 있던 것 같은 느낌이 들었습니다. 그 뭐냐... Bootstraping? Bagging도 있지 않나? 꺼내놓고 보니 세 단어 모두 들어보기만 했지 제대로 의미를 모르고 있었습니다. 그래서 이번 글에서는 저도 공부할 겸 겸사겸사 각 단어의 의미에 대해 알아보려 합니다.

구글링 결과 비슷한 고민을 한 사람이 있음을 Quora에서 알게 되었습니다. [What is the difference between boost, ensemble, bootstrap and bagging?](<https://www.quora.com/What-is-the-difference-between-boost-ensemble-bootstrap-and-bagging>)이라는 글이었습니다. 해당 글의 답변을 바탕으로 정리해봅시다.

### Boosting
Boosting은 ensemble 방법으로 weak learner 여러 개를 합쳐 strong learner를 만드는 방법입니다. Bias와 variance를 줄이는 것에 사용됩니다. Weighted majority voting(분류 문제)나 weighted sum(회귀 문제)를 통해 해당 작업을 진행하며 Ada boost와 Gradient boosting이 대표적인 방법이라고 합니다.

### Bootstrapping
복원 추출 방법으로, 전체 데이터의 일부(보통 2/3)만을 추출하고 나머지는 out-of-bag instance라고 칭하며 사용하지 않는 방법입니다. Train 과정에서 out-of-bag instance가 제외되기 때문에 따로 cross validation할 필요 없이 out-of-bag instance로 evaluation을 할 수 있습니다.

### Bagging
Bootstrap aggregating이라고도 불리며, variance를 줄여 과적합을 방지합니다. Bootstrapping으로 얻어진 모델을 여러 개 활용하여 equal weight vote를 시행하면 ensemble의 효과와 함께 좋은 모델을 만들 수 있게 됩니다.

### 출처
https://www.quora.com/What-is-the-difference-between-boost-ensemble-bootstrap-and-bagging