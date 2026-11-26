---
title: "RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning"
date: 2022-04-27 19:29:18 +0900
categories: [ML/DL, Reinforcement Learning]
tags: [English, reinforcement-learning, david-silver]
original_url: https://seungwooham.tistory.com/58
math: true
---

This article summarizes RL Course by David Silver - Lecture 1: Introduction to Reinforcement Learning.

![](/assets/img/posts/58/01.webp){: width="960" height="720" }

Reinforcement learning has no supervisor. The reward signal guides the agent if things are going right or wrong. The feedback is delayed. Some feedback takes hours to obtain, such as a game of Go. The data that the agent receives is sequential. So the action of the agent affects the information it receives.

![](/assets/img/posts/58/02.webp){: width="960" height="720" }

The reward hypothesis assumes a properly planned reward will lead the agent to the goal. By maximizing the expectation of accumulated reward, the agent can achieve its goal.

![](/assets/img/posts/58/03.webp){: width="960" height="720" }

To achieve a long-term goal, the agent may sacrifice immediate reward. We have to study to succeed in the exam, not play the game, even if playing the game gives us immediate pleasure. The agent learns this lesson by trial and error.

![](/assets/img/posts/58/04.webp){: width="960" height="720" }

Every time step, the agent observes the environment by itself. And every time step, the environment receives action from the agent and emits the observation.

![](/assets/img/posts/58/05.webp){: width="960" height="720" }

History is the whole sequence of the agent\'s experiences and behavior. And that history produces a state through some function.

![](/assets/img/posts/58/06.webp){: width="960" height="720" }

The agent usually cannot observe the whole information of the environment. Some aspects of the environment\'s information may be just noise. Therefore, the state is the agent\'s internal representation. In other words, $$S^{e}_{t} \not\equiv  S^{a}_{t}$$

![](/assets/img/posts/58/07.webp){: width="960" height="720" }

If the state has a Markov property, the present contains all useful information about the past.

![](/assets/img/posts/58/08.webp){: width="960" height="720" }

If the world is fully observable, the agent\'s state and the environment\'s state are the same. This is a Markov decision process(MDP) with a Markov property and fully observable characteristics.

![](/assets/img/posts/58/09.webp){: width="960" height="720" }

There is a different problem set for the partially observable environment, which we call the partially observable Markov decision process. If the environment is partially observable, the agent should estimate the environment by its belief or model.

![](/assets/img/posts/58/10.webp){: width="960" height="720" }

In RL, an agent may include one or more of these components: a policy(behavior), a value function, and a model. The value function estimates the value of the current state. Model is the way the agent understands the environment.

![](/assets/img/posts/58/11.webp){: width="960" height="720" }

Policy maps the state to action. Some policies are deterministic, while others are not. We use the character $$\pi$$ to represent a policy.

![](/assets/img/posts/58/12.webp){: width="960" height="720" }

The value function is the prediction of future rewards. We discount the future reward by $$\gamma$$, because it is mathematically efficient, and an immediate reward is usually better.

![](/assets/img/posts/58/13.webp){: width="960" height="720" }

P is the transition probability, and R is an estimated reward. In a model-based agent, this model is the basis of policy and value function because it helps the agent predict the future.

![](/assets/img/posts/58/14.webp){: width="960" height="720" }

This is the form of policy function, where there is an action for every state.

![](/assets/img/posts/58/15.webp){: width="960" height="720" }

This is an example of a value function where every state has its estimation of the value.

![](/assets/img/posts/58/16.webp){: width="960" height="720" }

The agent\'s model may be imperfect. So the model\'s transition updates as the agent explores the environment.

![](/assets/img/posts/58/17.webp){: width="960" height="720" }

We can categorize the RL agent as value-based, policy-based, and actor-critic. Actor-critic uses both value function and policy.

![](/assets/img/posts/58/18.webp){: width="960" height="720" }

This is another way to categorize RL agents: model-free and model-based.

![](/assets/img/posts/58/19.webp){: width="960" height="720" }

The taxonomy can be visualized as above.

![](/assets/img/posts/58/20.webp){: width="960" height="720" }

There is a big difference between reinforcement learning and planning. RL does not know the environment, and the agent should explore by updating its information about the environment. However, planning has full details on the environment. This means the agent can simulate the exact future.

Reinforcement learning is like trial-and-error learning. The agent should discover a good policy from its experiences of the environment, while planning is a full simulation, just like Dr. Strange.

![](/assets/img/posts/58/21.webp){: width="960" height="720" }

There are two different behaviors of the agent: exploration and exploitation. Exploration finds more information about the environment. Exploitation exploits known information to maximize reward.

![](/assets/img/posts/58/22.webp){: width="960" height="720" }

In the prediction stage, we evaluate the future with our known policy. In the control stage, we optimize the future by finding the best policy.

![](/assets/img/posts/58/23.webp){: width="960" height="720" }

The optimal value function $$v_{*}$$ is the maximum value function over all possible policies. The policy guides behavior in every possible state.
