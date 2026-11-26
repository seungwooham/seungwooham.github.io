---
title: "RL Course by David Silver - Lecture 4: Model Free Prediction"
date: 2022-08-04 15:39:57 +0900
categories: [ML/DL, Reinforcement Learning]
tags: [English, reinforcement-learning, david-silver, model-free-prediction]
original_url: https://seungwooham.tistory.com/65
math: true
---

This article summarizes RL Course by David Silver - Lecture 4: Model Free Prediction. This chapter will mainly discuss **how to predict the value of each** **state**. **Monte Carlo** and **Temporal Difference** are the two main streams of algorithms.

![](/assets/img/posts/65/01.webp){: width="3000" height="2250" }

Chapter 3 was about known MDP. Here we deal with an unknown MDP. Model-free prediction updates the value function of unknown MDP, and model-free control updates the policy based on value.

## Monte Carlo Reinforcement Learning

![](/assets/img/posts/65/02.webp){: width="3000" height="2250" }

First, we will check the Monte Carlo(MC) algorithm. MC algorithm does not need MDP knowledge (transition and rewards). But it can only be applied when the environment is episodic. MC uses empirical mean return instead of expected return. It does not bootstrap, because it uses the actual return $$G_{t}$$ observed until the end of the episode (an unbiased estimate of $$v_{\pi}(S_{t})$$) as its target.

![](/assets/img/posts/65/03.webp){: width="3000" height="2250" }

When we incrementally calculate the mean by using $$1/k$$ as a coefficient of $$x_{k}- \mu_{k-1}$$, we do not need to store the whole past value of $$x_{k}$$.

![](/assets/img/posts/65/04.webp){: width="3000" height="2250" }

Changing the $$k$$ to the number of visit counts $$N(S_{t})$$, the update is transformed to $$V(S_{t}) + (1/N(S_{t}))(G_{t}-V(S_{t}))$$. $$1/N(S_{t})$$ can be changed into a constant $$\alpha$$ for non-stationary problems.

![](/assets/img/posts/65/05.webp){: width="3000" height="2250" }

The plane in the slide above results from the value function in Blackjack by MC.

## Temporal-Difference Learning

![](/assets/img/posts/65/06.webp){: width="3000" height="2250" }

The temporal Difference(TD) algorithm learns directly from the action and reward. Like MC, TD does not need knowledge of the MDP. TD can also learn from the incomplete episodes by bootstrapping. TD updates a guess using a guess.

![](/assets/img/posts/65/07.webp){: width="3000" height="2250" }

The major difference between MC and TD is how they update the value of $$S_{t}$$. MC updates the value using actual return $$G_{t}$$, and TD updates the value by estimated return $$R_{t+1} + \gamma V(S_{t+1})$$. $$R_{t+1} + \gamma V(S_{t+1})$$ is also a guess, but at least it has data one step further.

![](/assets/img/posts/65/08.webp){: width="3000" height="2250" }

This is an example of going home from work. Every time the event happens, the expectation is updated.

## Difference between MC and TD: 1. Update Process

![](/assets/img/posts/65/09.webp){: width="3000" height="2250" }

Recalling the example above, the update of MC is done by the actual outcome. However, TD updates the estimation every single time with a new prediction.

![](/assets/img/posts/65/10.webp){: width="3000" height="2250" }

MC needs to end the whole episode to update. TD, on the other hand, needs only one more step. MC gives unbiased estimate of $$v_{\pi}(S_{t})$$.

![](/assets/img/posts/65/11.webp){: width="3000" height="2250" }

However, the TD target $$R_{t+1}+\gamma V(S_{t+1})$$ is a biased estimate of true $$v_{\pi}(S_{t})$$. So in the aspect of bias, MC has an advantage over TD. On the other hand, TD has a much lower variance.

## Difference between MC and TD: 2. When There is Only a Limited Sample

![](/assets/img/posts/65/12.webp){: width="3000" height="2250" }

MC and TD converge when there is infinite experience. How about when the experience is limited to batch size K?

![](/assets/img/posts/65/13.webp){: width="3000" height="2250" }

Let us figure it out with the example above. The left sequence shows the experience of the agent. The right figure is the Markov model estimated from these 8 episodes (the maximum likelihood Markov model), not the true environment, which is unknown.

![](/assets/img/posts/65/14.webp){: width="3000" height="2250" }

When MC and TD try to figure out the value of A, MC converges to a solution with minimum mean-squared error, and TD(0) converges to the solution of the maximum likelihood Markov model.

- TD exploits Markov property. It is usually more efficient in Markov environments.
- MC does not exploit Markov property. Usually more effective in non-Markov environments.

## Difference between MC and TD: 3. How They Backup, Bootstrap, and Sample

![](/assets/img/posts/65/15.webp){: width="3000" height="2250" }

This is how MC makes a backup.

![](/assets/img/posts/65/16.webp){: width="3000" height="2250" }

And this is how TD makes a backup. In this case, TD(0).

![](/assets/img/posts/65/17.webp){: width="3000" height="2250" }

Lastly, the DP backup. Dynamic programming has full knowledge of transition probability and reward so that it can calculate the expectation.

![](/assets/img/posts/65/18.webp){: width="3000" height="2250" }

Bootstrap contains estimates in an update process. In this point of view, MC does not bootstrap, but DP and TD bootstrap. In other words, bootstrapping methods use shallower backups than MC. In sampling, MC and TD do samples. But DP does not because it knows the whole transition data of MDP.

![](/assets/img/posts/65/19.webp){: width="3000" height="2250" }

It shows the overall position of each algorithm; the Y-axis shows whether the algorithm uses sampling, and the X-axis shows whether it uses bootstrapping.

## n-step TD

![](/assets/img/posts/65/20.webp){: width="3000" height="2250" }

TD algorithm can be modified by using n-step TD. For example, maximum n means it uses MC, and with n = 1, the TD becomes a 1-step TD, which is TD(0).

![](/assets/img/posts/65/21.webp){: width="3000" height="2250" }

The $$\lambda$$-return $$G_{t}^{\lambda}$$ averages all n-step returns $$G_{t}^{(n)}$$ with geometrically decaying weights $$(1-\lambda)\lambda^{n-1}$$. Forward-view TD($$\lambda$$) updates the value function toward $$G_{t}^{\lambda}$$. Because the geometric weighting is memoryless, TD($$\lambda$$) can be computed at the same cost as TD(0) (using the eligibility traces below), but the forward view has the drawback that it needs a full memory of future returns. It also requires ending the whole episode.

![](/assets/img/posts/65/22.webp){: width="3000" height="2250" }

To handle this drawback, we use the eligibility traces. The eligibility decreases geometrically because it is multiplied by $$\gamma\lambda$$ at each time step and incremented by one if the agent visits the state.

![](/assets/img/posts/65/23.webp){: width="3000" height="2250" }

Using the eligibility trace, we can change forward view TD($$\lambda$$) to backward view TD($$\lambda$$).

## The Relationship Between Forward and Backward TD

![](/assets/img/posts/65/24.webp){: width="3000" height="2250" }

The theorem shows the sum of the offline updates is identical for forward-view and backward-view TD($$\lambda$$).

![](/assets/img/posts/65/25.webp){: width="3000" height="2250" }

The chart summarizes the result.
