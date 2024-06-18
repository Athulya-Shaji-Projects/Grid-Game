# Grid-Game
import numpy as np

# global variables
BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
START = (1, 0)
DETERMINISTIC = True


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 10
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def nxtPosition(self, action):
        """
        action: "north", "south", "west", "east"
        -------------
        0 | 1 | 2| 3| 4
        1 |
        2 |
        3 |
        4 |
        return next position
        """
        if self.determine:
            if action == "north":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "south":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "west":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            # if next state legal
            if self.state == (1,3):
                nxtState = (3,3)
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (3,2) and nxtState != (2,2) and nxtState != (2,3) and nxtState != (2,4):
                        return nxtState
            return self.state

    def showBoard(self):
        self.board[self.state] = 1
        self.board[2,2] = -1
        self.board[2,3]= -1 
        self.board[2,4] = -1
        self.board[3,2]= -1
        self.board[4,4] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


# Agent of player

class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["north", "south", "west", "east"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3
        self.discount = 0.5

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if i== 1 and j == 3:
                    self.state_values[(i, j)] = 5
                else:
                    self.state_values[(i, j)] = -1
        

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds=10):
        i = 0
        ep = 0
        q_rewards = []       
        print(rounds)
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd or len(self.states) > 50: 
                # back propagate
                reward = self.State.giveReward()
                # explicitly assign end state to reward values
                self.state_values[self.State.state] = reward  # this is optional
                print("Game End Reward", reward)
                print(self.states)
                self.states.reverse()  
                # assigning rewards to each state          
                for s in self.states:
                    reward = self.state_values[s] + self.lr * (reward - self.state_values[s]) 
                    self.state_values[s] = round(reward, 3)
                self.states.reverse()
                q_reward = 0
                # calculating the cumulative reward of the current episode
                for j in range(0,len(self.states)):
                    r =  self.state_values[self.states[j]]           
                    q_reward = q_reward + (self.discount**j)*(r)
                # storing the cumulative reward of all the episodes in a lis
                q_rewards.append(q_reward)
                # stop training if average cumulative reward is greater than 10 over 30 consecutive episodes
                if np.mean(q_rewards)> 10:
                    ep = ep+1
                    if ep > 30:
                        break
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append(self.State.nxtPosition(action))
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    ag.play(100)
    print(ag.showValues())
#    st=State()
#    st.showBoard()
