import random
import numpy as np
from SarsaAlgorithm import Sarsa


'''
In this programing exercise we are going to solve the Cliff Walking problem explained in Example 6.6 page 132 of the textbook.
You need to apply Q-learning and SARA to this problem and show that:
1- SARSA converges to the blue path and Q-learning converges to the red path (shortest path). Explain why is that?
2- Generate the plot of sum of the rewards as a function of episodes. Explain why Q-leaning converges to lower average rewards even though it can find the optimal path?
3- For both methods gradually reduce the 𝜖𝜖 (in epsilon-greedy) and show that both algorithms converge to optimal path and explain why.

From book:
This is a standard undiscounted, episodic task, with start and goal states, and the usual actions causing movement up, down,right, and left.
Reward is -1 on all transitions except those into the region marked “The Cliff.” Stepping into this region incurs a reward of -100 and sends the agent instantly back to the start.
e-greedy action selection, e = 0.1.

a) The robot can move in 4 directions (4 straight).
b) Start location is 3,0
c) Goal location is 3,11
d) 1 means occupied and 0 means free
'''
class QLearning(Sarsa):

    def run(self,debug_state=-1):
        rewards_sum = []
        i=0
        while i< self.max_episodes:
            episodeRewardSum = 0
            i += 1

            if self.reduceEpsilon and i>2 and i%2==0:
                self.epsilon = self.epsilon * .75
                print("Epsilon reduced to: ", self.epsilon)

            # Start in start state
            currentState = self.flatten(self.start[0],self.start[1])

            # Loop for each step of episode:
                # Take action A, observe R, S0
                # Choose A0 from S0 using policy derived from Q (e.g., "-greedy)
                # Q(S,A)⇤Q(S,A) + alpha*[R + gamma*argmax[Q(S1,A1)] − Q(S,A)]
                # S⇤S1; A⇤A1;
            # until S is terminal
            while not self.isGoal(currentState):

                # Choose A from S using policy derived from Q (e.g., "e-greedy)
                currentAction = self.getActionFromStateDerivedQ(currentState)

                # Take action A, observe R, S'
                state1 = self.getNextState(currentState,currentAction)
                reward1 = self.rewards_matrix.flat[state1]

                greedyAction = np.argmax(self.Q[state1])

                self.Q[currentState,currentAction] = self.Q[currentState,currentAction] + self.alpha*(reward1 + self.gamma*self.Q[state1, greedyAction] - self.Q[currentState,currentAction])
                
                if self.isObstacle(state1):
                    state1 = self.flatten(self.start[0],self.start[1])
                
                currentState = state1
                episodeRewardSum += reward1

            rewards_sum.append(episodeRewardSum)

        return rewards_sum
        print("Completed run() on episodes")
       

    
