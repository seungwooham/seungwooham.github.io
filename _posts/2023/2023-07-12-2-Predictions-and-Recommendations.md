---
title: "2. Predictions and Recommendations"
date: 2023-07-12 10:22:43 +0900
categories: [ML/DL, Recommender System]
tags: [Korean, recommender-system, prediction]
original_url: https://seungwooham.tistory.com/103
---

본 글은 University of Minnesota의 Coursera 수업 [Introduction to Recommender Systems: Non-Personalized and Content-Based](https://www.coursera.org/learn/recommender-systems-introduction) 중 Predictions and Recommendations 강의의 요약입니다.

## 1. Learning Objectives

- Prediction/recommendation 구분
- Organic/explicit presentation 구분

## 2. Prediction/recommendation 구분

Prediction: 당신이 특정 item을 얼마나 좋아할지 예측

- Pro: help quantify item
- Con: 틀린 값을 줄 수도 있음

Recommendation: Suggestion for items, 보다 큰 item 집단에서 top-n list로 주어짐

- Pro: 좋은 선택지들을 나열해줌
- Con: 소비자의 탐험 기회를 앗아감

## 3. Organic/explicit presentation

Organic presentation

- 말 그대로 자연스러운 presentation
- 강한 압박을 기반으로 한 recommendation이 아니라 더 soft 함
- Top-n과 soft presentation의 조합으로 de-emphasize한 prediction/recommendation을 실행

Explicit presentation

- Organic의 case와 반대라고 보면 됨
