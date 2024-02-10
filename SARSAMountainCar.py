import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time

# SARSA algorithm
class SARSA:
    def __init__(self, env, discount = 1, stepSize = 0.1, epsilon = 0.1, maxIterations = 1000):
        self.env = env
        self.gamma = discount
        self.alpha = stepSize
        self.epsilon = epsilon
        self.maxIterations = maxIterations
        self.numPosBins = 20
        self.numVelBins = 20
        self.posBins = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], self.numPosBins)
        self.velBins = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], self.numVelBins)
        self.maxStates = self.QuantizedStates(self.env.observation_space.high)
        # initial random policy
        self.policy = np.random.rand(self.maxStates, self.env.action_space.n)
        self.policy = self.policy / np.sum(self.policy, axis = 1)[:, np.newaxis]
        # initial random Q Value function
        self.qValueFunction = np.zeros((self.maxStates, self.env.action_space.n))
        self.optimalPolicy = np.zeros((self.maxStates, self.env.action_space.n))

    def QuantizedStates(self, state):
        pos, vel = state
        binPos = np.digitize(pos, self.posBins, right = False)
        binVel = np.digitize(vel, self.velBins, right = False)
        state = binPos + binVel * self.numPosBins
        return state

    def StepsPerEpisodeGenerator(self):
        actions = np.arange(self.env.action_space.n)
        state, _ = self.env.reset()
        state = self.QuantizedStates(state)
        action = np.random.choice(actions, p = self.policy[state])
        while True:
            nextState, reward, terminated, truncated, info = self.env.step(action)
            nextState = self.QuantizedStates(nextState)
            nextAction = np.random.choice(actions, p = self.policy[nextState])
            yield state, action, reward, nextState, nextAction, terminated, truncated
            if terminated or truncated:
                break
            state = nextState
            action = nextAction

    def PolicyEvaluation(self):
        itr = self.StepsPerEpisodeGenerator()
        while True:
            state, action, reward, nextState, nextAction, terminated, truncated = next(itr)
            if truncated or terminated:
                break
            self.qValueFunction[state, action] += self.alpha * (reward + self.gamma * self.qValueFunction[nextState, nextAction] - self.qValueFunction[state, action])
        return self.qValueFunction, terminated

    def eGreedification(self):
        for state in range(self.maxStates):
            self.policy[state, :] =  self.epsilon / self.env.action_space.n
            maximumPoints = np.where(self.qValueFunction[state, : ] ==  np.max(self.qValueFunction[state, :]))
            self.policy[state, maximumPoints[0]] = (1 - self.epsilon) / len(maximumPoints[0]) + self.epsilon / self.env.action_space.n
        return self.policy

    def PolicyIteration(self):
        terminatedCount = 0
        for i in range(self.maxIterations):
            if i % 100 == 0:
                print('Episode no : ', i)
            _, terminated = self.PolicyEvaluation()
            self.eGreedification()
            if terminated:
                terminatedCount += 1
        print('Terminated count : ', terminatedCount)
        self.optimalPolicy = self.policy
        return 0
    
    def testPolicy(self):
        env = gym.make('MountainCar-v0', render_mode = 'human', max_episode_steps=400)
        actions = np.arange(env.action_space.n)
        state, _ = env.reset()
        state = self.QuantizedStates(state)
        cumReward = 0
        while True:
            action = np.random.choice(actions, p = self.optimalPolicy[state])
            nextState, reward, terminated, truncated, info = env.step(action)
            nextState = self.QuantizedStates(nextState)
            cumReward += reward
            if terminated or truncated:
                time.sleep(1)
                break
            state = nextState
        print('Cummulative reward : ', cumReward)
        # cummulative reward is -145
env = gym.make('MountainCar-v0', max_episode_steps = 200)
obj = SARSA(env, maxIterations = 1500)
obj.PolicyIteration()
obj.testPolicy()

