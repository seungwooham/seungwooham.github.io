---
title: "RL Course by David Silver - Lecture 3: Planning by Dynamic Programming"
date: 2022-05-18 21:57:10 +0900
categories: [ML/DL, Reinforcement Learning]
tags: [English, reinforcement-learning, david-silver, dynamic-programming]
original_url: https://seungwooham.tistory.com/61
math: true
---

This article summarizes RL Course by David Silver - Lecture 3: Planning by Dynamic Programming. This chapter will discuss **policy evaluation, policy iteration, and value iteration**. Each concept has an important role in reinforcement learning.

## Introduction

![](/assets/img/posts/61/01.webp){: width="960" height="720" }

Before we start, what is dynamic programming? A dynamic problem has a sequential or temporal component. Programming is optimizing a program(in here, policy). Dynamic programming breaks down the complex problem into subproblems and combines the solution.

![](/assets/img/posts/61/02.webp){: width="960" height="720" }

We need two requirements for dynamic programming: **1. The optimal solution can be decomposed into subproblems. 2. Subproblems recur many times.**

The Markov decision process satisfies both properties. Bellman equation itself is a recursive decomposition. The value function is stored and reused over time.

![](/assets/img/posts/61/03.webp){: width="960" height="720" }

There are two stages in an MDP: **prediction and control**. In the prediction stage, it predicts the value function given the policy. In the control stage, it finds the optimal value function and the optimal policy given only the MDP.

## Iterative Policy Evaluation

![](/assets/img/posts/61/04.webp){: width="960" height="720" }

An iterative policy evaluation uses **Bellman expectation** backup to **evaluate a given policy** $$\pi$$. The process is progressed synchronously. At each iteration $$k+1$$, we update all states from the values of iteration $$k$$. It updates $$v_{k+1}(s)$$ from $$v_{k}(s')$$.

![](/assets/img/posts/61/05.webp){: width="960" height="720" }

The diagram shows how the state value is updated by iterative policy evaluation.

![](/assets/img/posts/61/06.webp){: width="960" height="720" }

This small grid world is undiscounted. The white room means a non-terminal state. The black room means a terminal state. Each movement has a reward of -1 until the terminal state is reached. The agent follows a uniform random policy.

When the iteration reaches $$k=2$$, the greedy policy is updated, and it certainly behaves better than a random policy.

![](/assets/img/posts/61/07.webp){: width="960" height="720" }

When $$k=3$$, an optimal policy is already reached. By iterative policy evaluation with non-optimal policy, we identified an optimal policy.

![](/assets/img/posts/61/08.webp){: width="960" height="720" }

Given a policy $$\pi$$, when we evaluate the value function with the policy $$\pi$$, acting greedily with respect to $$v_{\pi}$$ gives an improved policy. In a small grid world example, we reached optimal policy in 3rd iteration, but it usually takes longer than that. This process of **policy iteration always converges to** $$\pi^{*}$$.

## Policy Iteration

![](/assets/img/posts/61/09.webp){: width="960" height="720" }

The process we saw in the policy evaluation can be generalized as a **policy iteration**. **Policy iteration repeats the policy evaluation step and policy improvement step.**

![](/assets/img/posts/61/10.webp){: width="960" height="720" }

Does acting greedily always give us a better policy? Yes. Because a new policy $$\pi'$$ improves the value from any state $$s$$ over one step.

![](/assets/img/posts/61/11.webp){: width="960" height="720" }

If improvements stop, the Bellman optimality equation is satisfied, and therefore $$v_{\pi}(s) = v_{*}(s)$$ for all $$s\in S$$.

In practice, the stopping condition ($$\epsilon$$-convergence of value function, fixed number of iterations) is discussed.

![](/assets/img/posts/61/12.webp){: width="960" height="720" }

This property can be generalized to any policy evaluation algorithm and any policy improvement algorithm. So, reinforcement learning can be applied in various fields.

## Value Iteration

![](/assets/img/posts/61/13.webp){: width="960" height="720" }

We can induce the principle of optimality, which is quite obvious.

![](/assets/img/posts/61/14.webp){: width="960" height="720" }

If we know the solution to the subproblem, the value of the current state $$v_{*}(s)$$ can be found by one-step lookahead from the values of its successor states $$v_{*}(s')$$. The recursive one-step lookahead becomes a solution for all states. In value iteration, **the value of a state is updated with the maximum one-step lookahead value over all actions, i.e., the immediate reward plus the discounted value of the successor states.**

![](/assets/img/posts/61/15.webp){: width="960" height="720" }

This is the problem of finding the shortest path to goal g. And each state\'s value is calculated recursively, from the state near the goal to the state far from the goal.

![](/assets/img/posts/61/16.webp){: width="960" height="720" }

With a **Bellman optimality backup**, we can find an optimal value of each state. There is no explicit policy. It just looks one step ahead for the best immediate action, which provides an agent with a greedy update.

![](/assets/img/posts/61/17.webp){: width="3000" height="2250" }

​This diagram shows the value iteration process. Where $$v_{k+1}(s)$$ is updated from the values $$v_{k}(s')$$ of the successor states $$s'$$ by the action which gives $$s$$ the maximum value.



![](/assets/img/posts/61/18.webp){: width="960" height="720" }

We can divide the problem into prediction and control. And the algorithm can be divided into iterative policy evaluation, policy iteration, and value iteration. However, acquiring the action-value function is more complex because it is defined over state-action pairs.
