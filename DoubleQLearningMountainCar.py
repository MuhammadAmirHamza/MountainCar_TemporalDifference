import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time

# Expected SARSA algorithm
class DoubleQLearning:
    def __init__(self, env, discount = 1, stepSize = 0.1, epsilon = 0.2, maxIterations = 1000):
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
        self.qValueFunctionA = np.zeros((self.maxStates, self.env.action_space.n))
        self.qValueFunctionB = np.zeros((self.maxStates, self.env.action_space.n))
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
        while True:
            action = np.random.choice(actions, p = self.policy[state])
            nextState, reward, terminated, truncated, info = self.env.step(action)
            nextState = self.QuantizedStates(nextState)
            yield state, action, reward, nextState, terminated, truncated
            if terminated or truncated:
                break
            state = nextState

    def PolicyEvaluation(self):
        itr = self.StepsPerEpisodeGenerator()
        while True:
            state, action, reward, nextState, terminated, truncated = next(itr)
            if truncated or terminated:
                break
            
            # generate a toss
            randNum = np.random.rand()
            if randNum > 0.5:
                self.qValueFunctionA[state, action] += self.alpha * (reward + self.gamma * self.qValueFunctionB[nextState, np.argmax(self.qValueFunctionA[nextState, :])] - self.qValueFunctionA[state, action])
            else:
                self.qValueFunctionB[state, action] += self.alpha * (reward + self.gamma * self.qValueFunctionA[nextState, np.argmax(self.qValueFunctionB[nextState, :])] - self.qValueFunctionB[state, action])                         
            
        return self.qValueFunctionA, self.qValueFunctionB, terminated

    def eGreedification(self):
        qValueFunction = (self.qValueFunctionA + self.qValueFunctionB) / 2
        for state in range(self.maxStates):
            self.policy[state, :] =  self.epsilon / self.env.action_space.n
            maximumPoints = np.where(qValueFunction[state, : ] ==  np.max(qValueFunction[state, :]))
            self.policy[state, maximumPoints[0]] = (1 - self.epsilon) / len(maximumPoints[0]) + self.epsilon / self.env.action_space.n
        return self.policy

    def PolicyIteration(self):
        terminatedCount = 0
        for i in range(self.maxIterations):
            if i % 100 == 0:
                print('Episode no : ', i)
            _, _, terminated = self.PolicyEvaluation()
            self.eGreedification()
            if terminated:
                terminatedCount += 1
        print('Terminated count : ', terminatedCount)
        # greedification of optimal policy
        for state in range(self.maxStates):
            maximumPoints = np.where(self.policy[state, :] == np.max(self.policy[state, :]))[0]
            self.optimalPolicy[state, maximumPoints] = 1/len(maximumPoints)
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
obj = DoubleQLearning(env, maxIterations = 1500)
obj.PolicyIteration()

obj.testPolicy()

