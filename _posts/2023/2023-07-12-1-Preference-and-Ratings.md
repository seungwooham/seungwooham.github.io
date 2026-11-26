---
title: "1. Preference and Ratings"
date: 2023-07-12 09:50:35 +0900
categories: [ML/DL, Recommender System]
tags: [Korean, recommender-system, preference, rating]
original_url: https://seungwooham.tistory.com/102
---

본 글은 University of Minnesota의 Coursera 수업 [Introduction to Recommender Systems: Non-Personalized and Content-Based](https://www.coursera.org/learn/recommender-systems-introduction) 중 Preferences and Ratings 강의의 요약입니다.

이번 강의의 preference와 rating의 예시는 주로 MovieLens 데이터 세트를 기준으로 합니다.

## 1. 종류

Explicit(분명)한 ratings

- Star rating
- Review
- Vote(Like button, Ephemeral(단명하는) contents에서 더 자주 사용됨)

Implicit(암시적인)한 ratings

- Click
- Purchase
- Follow

Other Interfaces

- Continuous scales
- Pairwise preference(둘 중에 어떤 것이 더 낫다)
- Hybrid(Continuous + Never again)
- Temporary (Pandora 30-day suspension)

## 2. Explicit한 Ratings

언제 rating을 부여하는가?

- Consumption
- Memory
- Expectation

Difficulties with ratings

- reliability and accuracy(신뢰성과 정확성)
- Change of preference over time
- The meaning of the rating

## 3. Implicit한 Ratings

Reading time

- how long do they read or listen

Binary actions

- Click on link or not (나쁠 것이라 짐작하고 클릭 자체를 안함)
- Purchase (구매했더라도 싫어할 수도 있음)
- Follow/friend

## 4. Conclusion

- Recommenders는 user의 say와 do에서 preference를 배움
- Rating은 preference에 대한 explicit한 expression
- Implicit data는 큰 volume을 얻는다는 장점 존재
