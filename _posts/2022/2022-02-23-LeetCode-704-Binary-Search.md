---
title: "LeetCode 704. Binary Search"
date: 2022-02-23 17:26:07 +0900
categories: [ML/DL, Algorithm]
tags: [English, leetcode, binary-search]
original_url: https://seungwooham.tistory.com/26
---

## Problem :

Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

## Example :

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

```
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

## Constraints :

- 1 &lt;= nums.length &lt;= 10<sup>4</sup>
- -10<sup>4</sup> &lt; nums[i], target &lt; 10<sup>4</sup>
- All the integers in nums are **unique**.
- nums is sorted in ascending order.

## Solution :

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        idx = 0
        max_ = len(nums)-1
        min_ = 0
        while (nums[idx] != target):
            if (max_-min_)<=1:
                if (nums[max_]==target):
                    idx=max_
                elif (nums[min_]==target):
                    idx=min_
                else :
                    idx=-1
                break
            else :
                idx = (max_+min_)//2
                if nums[idx] > target: max_=idx
                else : min_=idx
        return idx
```

An official solution from LeetCode

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = left + (right - left) // 2
            if nums[pivot] == target:
                return pivot
            if target < nums[pivot]:
                right = pivot - 1
            else:
                left = pivot + 1
        return -1
```

## Link to the Source :

[https://leetcode.com/problems/binary-search/](https://leetcode.com/problems/binary-search/)
