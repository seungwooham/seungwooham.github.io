---
title: "RL Course by David Silver - Lecture 2: Markov Decision Processes"
date: 2022-05-03 23:24:09 +0900
categories: [ML/DL, Reinforcement Learning]
tags: [English, reinforcement-learning, david-silver, markov-decision-process]
original_url: https://seungwooham.tistory.com/59
math: true
---

This article summarizes RL Course by David Silver - Lecture 2: Markov Decision Processes.

## Markov Property

![](/assets/img/posts/59/01.webp){: width="840" height="630" }

When we assume all relevant information about the past is in the present state, that state has a Markov property.

![](/assets/img/posts/59/02.webp){: width="840" height="630" }

The state transition matrix $$P$$ contains all information about transition probability between states.

## Markov Process

![](/assets/img/posts/59/03.webp){: width="840" height="630" }

The sequence of states with the Markov property is a Markov process (or Markov chain). Markov process contains two elements: States and Probability matrix.

## Markov Reward Process

![](/assets/img/posts/59/04.webp){: width="840" height="630" }

The Markov reward process is a Markov process (or Markov chain) with values. It adds two elements: a reward function, which is an expectation of reward, and a discount factor.

![](/assets/img/posts/59/05.webp){: width="840" height="630" }

Return is discounted by a discount factor $$\gamma$$. When $$\lambda$$ is close to 0, it means the myopic evaluation. $$\lambda$$ close to 1 means far-sighted evaluation. Why do we need a discount factor?

1\. It is mathematically convenient to discount rewards.

2\. It avoids infinite returns in cyclic Markov processes.

3\. Uncertainty about the future may not be fully represented.

4\. If the reward is financial, immediate rewards may earn more interest than delayed rewards.

5\. Animal/human behavior shows a preference for immediate reward.

![](/assets/img/posts/59/06.webp){: width="840" height="630" }

The value function $$v(s)$$ gives the long-term value of state $$s$$; in other words, it is the expected return starting from state $$s$$.

![](/assets/img/posts/59/07.webp){: width="840" height="630" }

Here, we can check the left-top cornered circle with the value of -1 because the evaluation is so myopic that it only counts the immediate reward.

![](/assets/img/posts/59/08.webp){: width="840" height="630" }

Bellman Equation for MRP has two parts: immediate reward $$R_{t+1}$$ and the discounted value of the successor state.

![](/assets/img/posts/59/09.webp){: width="840" height="630" }

The graphical representation of the Bellman equation can be written as above.

![](/assets/img/posts/59/10.webp){: width="840" height="630" }

The equation can be expressed in the matrix form, where $$n$$ represents a total number of possible states.

![](/assets/img/posts/59/11.webp){: width="840" height="630" }

The Bellman equation is a linear equation; thus it can be solved directly.

## Markov Decision Process

![](/assets/img/posts/59/12.webp){: width="840" height="630" }

The difference between the Markov decision process and the Markov reward process is the existence of actions.

![](/assets/img/posts/59/13.webp){: width="840" height="630" }

The action of MDP can be generalized into policy. The policy $$\pi$$ is a distribution over actions given states.

![](/assets/img/posts/59/14.webp){: width="840" height="630" }

Given a policy $$\pi$$, when we sum over all actions $$a$$ weighted by $$\pi(a \mid s)$$, the MDP is transformed into an MRP.

![](/assets/img/posts/59/15.webp){: width="840" height="630" }

There are two value functions: The state-value function and the action-value function. **State-value function is an expected return starting from state $$s$$ and then following policy $$\pi$$. The action-value function is the expected return starting from state $$s$$, taking action $$a$$, and following policy $$\pi$$.**

![](/assets/img/posts/59/16.webp){: width="840" height="630" }

The action-value function can be decomposed into immediate reward and discounted action-value of successor state-action pair.

![](/assets/img/posts/59/17.webp){: width="840" height="630" }

The state-value function and the action-value function have connectivity.

![](/assets/img/posts/59/18.webp){: width="840" height="630" }

The connectivity between two functions can be expressed vice-versa.

![](/assets/img/posts/59/19.webp){: width="840" height="630" }

Combining these two relationships, the Bellman expectation equation for $$v_{\pi}$$ can be transformed as above. The equation contains the state-value function and policy $$\pi$$.

![](/assets/img/posts/59/20.webp){: width="840" height="630" }

The transformation can also be applied to the Bellman expectation equation for $$q_{\pi}$$.

![](/assets/img/posts/59/21.webp){: width="840" height="630" }

Now we define the optimal value function. The optimal state-value function is the maximum value function over all policies. This means the maximum value function chooses argmax policy $$\pi$$ for its value. This is the same for the optimal action-value function.

![](/assets/img/posts/59/22.webp){: width="840" height="630" }

The optimal policy has to maintain the characteristics above, which is quite trivial. Furthermore, the optimal policy has to be better than or equal to every other policy in every state. And the optimal policy achieves the optimal value function. Lastly, the optimal policy achieves the optimal action-value function.

![](/assets/img/posts/59/23.webp){: width="840" height="630" }

The optimal policy found by maximizing over $$q_{*}(s,a)$$ is a one-hot action vector. There is always such a deterministic optimal policy for any MDP, and figuring out the optimal action-value function leads to the optimal policy.

![](/assets/img/posts/59/24.webp){: width="840" height="630" }

The optimal value functions are recursively related by the Bellman optimality equations.

![](/assets/img/posts/59/25.webp){: width="840" height="630" }

This is another direction of recursively related Bellman optimality equations.

![](/assets/img/posts/59/26.webp){: width="840" height="630" }

Here the only difference from the Bellman expectation equation is the existence of the max function.

![](/assets/img/posts/59/27.webp){: width="840" height="630" }

The same max function can be applied to the action-value function.

![](/assets/img/posts/59/28.webp){: width="840" height="630" }

Thus the Bellman optimality equation is non-linear, and a closed-form solution does not exist in general. This is the difference from the expectation equations.
