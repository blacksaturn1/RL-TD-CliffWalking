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

    def __init__(self, world_maze,max_episodes=1,alpha=.5,epsilon=0.1,gamma=1,start= (3,0), goal=(3,11),rewards=[-1,-100],actions=(0,1,2,3), reduceEpsilon=True):
        '''
        rewards = [goal, space, obstacle]
        '''
        self.max_episodes=max_episodes
        self.reduceEpsilon=reduceEpsilon
        self.iterations=0
        self.world=np.asarray(world_maze)
        self.actions = actions #(0,1,2,3) up, right, down, left
        self.N_States = len(self.world[0])*len(self.world)
        self.Q = 0
        self.rewards_matrix = np.zeros( self.world.shape )
        self.rewards=rewards
        self.alpha = alpha
        self.epsilon = 0.1
        self.gamma=gamma
        self.start = start
        self.goal = goal
        self.simple=1 # Flag for only 4 actions
        self.initRewardMatrix()
        self.initQ()
    
        
    def checkQ(self):
        print(self.Q[0:])
        #print("Hello")
        return

    def getRandomState(self):
        # convert to flat indexing
        w = len(self.world)
        l = len(self.world[0])
        s = random.choices(range(0,l*w))
        return s[0]

    def getRandomAction(self):
        randomAction = random.randrange(0,len(self.actions),1)
        return randomAction

    def flatTo2DState(self,x):
        row = x // len(self.world[0])
        diff = x - (row * len(self.world[0]) ) 
        column = diff
        return row, column
    
    def flatten(self,r,c):
        flat = r*len(self.world[0]) + c
        return flat
    
    
    
    def isGoal(self,stateNum):
        gs = self.goal[0]*len(self.world[0]) + self.goal[1]
        # print('goal', gs)
        if stateNum == gs:
            return True
        return False

    def isObstacle(self, stateNum):
        '''
        if(self.isWall(stateNum)):
            return False
        '''
        r, c = self.flatTo2DState(stateNum)
        if self.world[r,c]==1:
            return True
        
        return False

    def initRewardMatrix(self):
        for x in range(self.N_States):
            if(self.isGoal(x)):
                self.rewards_matrix.flat[x]=0
            elif(self.isObstacle(x)):
                self.rewards_matrix.flat[x]=self.rewards[1]
            else:
                self.rewards_matrix.flat[x]=self.rewards[0]
        return

    def initQ(self,probs=[0,1,0]):
        '''
        Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except that Q(terminal , ·) = 0
        '''
        self.Q = 1*np.ones( (self.N_States,len(self.actions) ))
        # Set goal to 0
        flatGoalState = self.flatten(self.goal[0],self.goal[1])
        self.Q[flatGoalState,:]=0
        return
    
    

    def getNextState(self,state,action):
        r,c = self.flatTo2DState(state)
        r1=r
        c1=c
        if self.simple==1:
            if action==0: #up
                r1=r-1
            elif action==1: #right
                c1=c+1
            elif action==2: #down
                r1=r+1
            elif action==3: #left
                c1=c-1
        else:
            if action==0:
                r1=r-1
            elif action==1:#up-right
                r1=r-1
                c1=c+1
            elif action==2: #right
                c1=c+1
            elif action==3: #down-right
                r1=r+1
                c1=c+1
            elif action==4: #down
                r1=r+1
            elif action==5: #down-left
                #down-left
                r1=r+1
                c1=c-1
            elif action==6: #left
                c1=c-1
            elif action==7:  #up-left
                r1=r-1
                c1=c-1
        if(c1>=len(self.world[0]) or c1<0) or (r1>=len(self.world) or r1<0):
            return state

        newNextState = self.flatten(r1,c1)
        return newNextState


    def getNextStatesGivenActionProbability(self, state,action,probs=[0,1,0]):
        '''returns [(prob1,action1,s_prime)]'''
        probs_Actions=[]
        index=0        
        for p in probs:
            action1=action
            if(index==0):
                action1=action-1
            if(index==2):
                action1=action+1
            prob=p
            
            if self.simple==0:
                if(action1<0):
                    action1=7
                if(action1>7):
                    action1=0
            else:
                if(action1<0):
                    action1=3
                if(action1>3):
                    action1=0
            if(prob>0):
                s_prime = self.getNextState(state,action1)
                probs_Actions.append((action1,prob,s_prime))
            index+=1
        return probs_Actions


    def getActionFromStateDerivedQ(self,state):
        epsilonProbability=random.random()
        action=-1
        if epsilonProbability<=self.epsilon:
            action = self.getRandomAction()
        else:
            action=np.random.choice(np.flatnonzero(self.Q[state,:] == self.Q[state,:].max())) # breaks ties randomly
        return action


    def run(self,debug_state=-1):
        
        i=0
        while i< self.max_episodes:
            i += 1

            if self.reduceEpsilon and i>100 and i%100==0:
                self.epsilon = self.epsilon * .9
                print("Epsilon reduced to: ",self.epsilon)

            # Start in start state
            currentState = self.flatten(self.start[0],self.start[1])

            # Choose A from S using policy derived from Q (e.g., "e-greedy)
            currentAction = self.getActionFromStateDerivedQ(currentState)

            # Loop for each step of episode:
                # Take action A, observe R, S0
                # Choose A0 from S0 using policy derived from Q (e.g., "-greedy)
                # Q(S,A)⇤Q(S,A) + [R + gamma*Q(S1,A1) − Q(S,A)]
                # S⇤S1; A⇤A1;
            # until S is terminal
            while not self.isGoal(currentState):
                state1 = self.getNextState(currentState,currentAction)
                reward1 = self.rewards_matrix.flat[state1]
                
                action1 = self.getActionFromStateDerivedQ(state1)
                self.Q[currentState,currentAction] = self.Q[currentState,currentAction] +self.alpha*(reward1 + self.gamma*self.Q[state1,action1] - self.Q[currentState,currentAction])
                
                if self.isObstacle(state1):
                    state1 = self.flatten(self.start[0],self.start[1])
                
                currentState = state1
                currentAction = action1

        print("Completed run() on episodes")
       

    def plot_value_function_convergence(self,delta_list):
      x = np.arange(0.0, len(delta_list), 1)
      fig, ax = plt.subplots()
      ax.plot(x, delta_list)
      ax.set(xlabel='iterations', ylabel='delta',
          title='Convergence of Evaluate Value Function')
      ax.grid()
      fig.savefig("test.png")
      plt.show()
    
    

    def getPolicyQuivers(self):
        x_pos=[]
        y_pos=[]
        x_direct=[]
        y_direct=[]

        currentState = self.flatten(self.start[0],self.start[1])
        while not self.isGoal(currentState):
            actionTaken = self.Q[currentState,:].argmax()
            r,c = self.flatTo2DState(currentState)
            x_pos.append(c)
            y_pos.append(r)
            u,v=self.makeArrowFromAction(actionTaken)
            x_direct.append(u)
            y_direct.append(v)
            nextStateTaken = self.getNextState(currentState,actionTaken)  
            currentState = nextStateTaken
        
        return x_pos, y_pos, x_direct, y_direct
             

        # add arrow to quiver collection
    def makeArrowFromAction(self, action):
        '''Returns u,v'''
        u=0
        v=0
        if  self.simple ==1:
            if action==0:
                u=0
                v=-1
            elif action ==1:
                u=1
                v=0    
            elif action ==2:
                u=0
                v=1;    
            elif action ==3:
                u=-1
                v=0
        else:
            if action ==0:
                u=0;v=-1
            elif action ==1:
                u=1;v=-1;    
            elif action ==2:
                u=1;v=0;    
            elif action ==3:
                u=1;v=1;    
            elif action ==4:
                u=0;v=1;    
            elif action ==5:
                u=-1;v=1;    
            elif action ==6:
                u=-1;v=0;    
            elif action ==7:
                u=-1;v=-1;
        return u,v   
   
