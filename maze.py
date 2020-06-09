import numpy as np
from IPython.display import clear_output
import time

class GridWorld:
  def __init__(self, m, n):
    self.grid = np.zeros((m, n))
    self.m = m
    self.n = n
    self.stateSpace = [i for i in range(self.m*self.n - 1)]
    self.stateSpacePlus = [i for i in range(self.m*self.n)]
    self.actionSpace = {'U' : -self.m,
                        'D' : +self.m,
                        'L' : -1,
                        'R' : +1}
    self.actions = ['L', 'R', 'U', 'D']
    self.agentPos = 0
    self.blockedSquares = [2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 34, 30, 31, 32, 33, 34]
    self.addBlockedSquares()

  def isTerminalState(self, state):
    return state in self.stateSpacePlus and not state in self.stateSpace
  
  def isOffGrid(self, newState, oldState):
    if newState not in self.stateSpacePlus:
      return True
    elif newState % self.m == 0 and oldState % self.m == self.m - 1:
      return True
    elif newState % self.m == self.m - 1 and oldState % self.m == 0:
      return True
    else: return False

  def addBlockedSquares(self):
    for square in self.blockedSquares:
      x = square // self.m
      y = square % self.n
      self.grid[x][y] = 2

  def getAgentRowAndCol(self):
    x = self.agentPos // self.m
    y = self.agentPos % self.n
    return x, y

  def setState(self, state):
    x, y = self.getAgentRowAndCol()
    self.grid[x][y] = 0
    self.agentPos = state
    x, y = self.getAgentRowAndCol()
    self.grid[x][y] = 1
    return

  def step(self, action):
    resultingState = self.agentPos + self.actionSpace[action]

    if resultingState in self.blockedSquares:
      reward = -100
    elif self.isTerminalState(resultingState):
      reward = +100
    else: reward = 0

    if not self.isOffGrid(resultingState, self.agentPos):
      self.setState(resultingState)
      return resultingState, reward, self.isTerminalState(self.agentPos), None
    else:
      return self.agentPos, reward, self.isTerminalState(self.agentPos), None
    
  def reset(self):
    self.grid = np.zeros((self.m, self.n))
    self.addBlockedSquares()
    self.agentPos = 0
    return self.agentPos

  def render(self):
    print("----------------------------")
    for x in range(self.m):
      for y in range(self.n):
        if x == 0 and y == 0:
          print("S", end = '\t')
        elif x == self.m - 1 and y == self.n - 1:
          print("G", end = '\t')
        elif self.grid[x][y] == 0:
          print("_", end = '\t')
        elif self.grid[x][y] == 1:
          print("X", end = '\t')
        elif self.grid[x][y] == 2:
          print("|", end = '\t')
      print('\n')
    print("----------------------------")
    return
    

def maxAction(Q, obs, actions):
  values = np.array([Q[obs, action] for action in actions])
  idx = np.argmax(values)
  return actions[idx]

  if __name__ == "__main__":
  env = GridWorld(6, 6)
  eps = 1.0
  epsdecay = 0.9995
  epsmin = 0.01
  alpha = 0.1
  gamma = 0.99
  n_episodes = 50000

  Q = {}
  for state in env.stateSpacePlus:
    for action in env.actions:
      Q[state, action] = 0
  env.render()
  for episode in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    count = 0
    while not done:
      #clear_output(wait = True)
      #env.render()
      #time.sleep(0.5)
      p = np.random.random()
      action = np.random.choice(env.actions) if p < eps else maxAction(Q, obs, env.actions)
      obs_, reward, done, _ = env.step(action)
      action_ = maxAction(Q, obs_, env.actions)
      total_reward += reward
      Q[obs, action] = (1 - alpha)*Q[obs, action] + alpha*(reward + gamma*Q[obs_, action_])
      obs = obs_
    eps = eps*epsdecay if eps*epsdecay > epsmin else epsmin
    if episode % 5000 == 0:
      print(episode, total_reward)
    



  obs = env.reset()
  done = False
  while not done:
    clear_output(wait = True)
    env.render()
    time.sleep(0.7)
    action = maxAction(Q, obs, env.actions)
    obs_, reward, done, _ = env.step(action)
    obs = obs_
