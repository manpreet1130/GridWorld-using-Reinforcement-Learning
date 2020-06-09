import numpy as np


class GridWorld:
    def __init__(self, m, n, magicalSquares):
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n
        self.squares = magicalSquares
        self.stateSpace = [i for i in range(self.m*self.n - 1)]
        self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U' : -self.m,
                            'D' : self.m,
                            'L' : -1,
                            'R' : +1}
        self.actions = ['U', 'D', 'L', 'R']
        self.agentPos = 0

    def isTerminalState(self, state):
        return state in self.stateSpace and state in self.stateSpacePlus

    def addMagicalSquares(self):
        i = 2
        for square in self.squares:
            x = square // self.m
            y = square % self.n
            grid[x][y] = i
            i += 1
            x = self.squares[square] // self.m
            y = self.squares[square] % self.n 
            grid[x][y] = i
            i += 1
        return

    def getAgentRowAndCol(self):
        x = self.agentPos // self.m
        y = self.agentPos % self.n
        return x, y

    def isOffGrid(self, newState, oldState):
        if newState not in self.stateSpacePlus:
            return True
        elif newState % self.m == 0 and oldState % self.m == self.m - 1:
            return True
        elif newState % self.m == self.m - 1 and oldState % self.m == 0:
            return True       
        else: return False

    def setState(self, state):
        x, y = self.getAgentRowAndCol()
        self.grid[x][y] = 0
        self.agentPos = state
        x, y = self.getAgentRowAndCol()
        self.grid[x][y] = 1
        return

    def step(self, action):
        resultingState = self.agentPos + self.actionSpace[action]
        if resultingState in self.squares:
            resultingState = self.squares[resultingState]

        reward = -1 if not self.isTerminalState(resultingState) else 0

        if not self.isOffGrid(resultingState, self.agentPos):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(self.agentPos), None
        else:
            return self.agentPos, reward, self.isTerminalState(self.agentPos), None

    
    def reset(self):
        self.grid = np.zeros((self.m, self.n))
        self.agentPos = 0
        return self.agentPos


    def render(self):
        print("----------------------------")
        for r in range(self.m):
            for c in range(self.n):
                if self.grid[r][c] == 0:
                    print('-', end = '\t')
                elif self.grid[r][c] == 1:
                    print("X")
                elif self.grid[r][c] == 2:
                    print("Ain")
                elif self.grid[r][c] == 3:
                    print("Aout")
                elif self.grid[r][c] == 4:
                    print("Bin")
                elif self.grid[r][c] == 5:
                    print("Bout")
            print('\n')
        print("----------------------------")
        return              

def maxAction(Q, obs, actions):
    values = np.array([Q[obs, action] for action in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == "__main__":
    magicalSquares = {15 : 63, 54 : 12}
    env = GridWorld(9, 9, magicalSquares)

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.actions:
            Q[state, action] = 0

    eps = 1.0
    epsdecay = 0.998
    epsmin = 0.01
    alpha = 0.1
    gamma = 0.99
    n_episodes = 50000
    env.render()
'''
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.random.choice(env.actions) if np.random.random() < eps else maxAction(Q, obs, env.actions)
            obs_, reward, done, _ = env.step(action)
            action_ = maxAction(Q, obs_, env.actions)
            total_reward += reward
            Q[obs, action] = (1 - alpha)*Q[obs, action] + alpha*(reward + gamma*Q[obs_, action_])
            obs = obs_
            eps = eps*epsdecay if eps*epsdecay < epsmin else epsmin
        if episode % 5000 == 0:
            print("Total Reward : {}, Episode : {}".format(total_reward, episode))
'''        
