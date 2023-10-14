import random
from helper import Helper
import numpy as np
from matplotlib import colors
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
class Sarsa:

    def __init__(self, world_maze,max_episodes=1,alpha=.5,epsilon=0.1,gamma=1,start= (3,0), goal=(3,11),rewards=[-1,-100],actions=(0,1,2,3)):
        '''
        rewards = [goal, space, obstacle]
        '''
        self.max_episodes=max_episodes
        self.iterations=0
        self.world=np.asarray(world_maze)
        self.actions = actions #(0,1,2,3)#,4,5,6,7) # up, down, left, right, up-left, up-right, dwn-left, dwn-right
        
        self.N_States = len(self.world[0])*len(self.world)

        #self.Q = [dict() for x in range(self.N_States)]
        self.Q = 0
        self.rewards_matrix = np.zeros( self.world.shape )
        self.rewards=rewards
        #V = np.zeros(self.N_States)
        self.alpha = alpha
        self.epsilon = 0.1
        self.gamma=gamma
        self.start = start
        self.goal = goal
        
        #self.policy = (1/len(self.actions))*np.ones((self.N_States,len(self.actions)))
        #self.value_function = np.zeros( self.world.shape )

        self.convergenceTrack = []
        self.convergenceTolerance = 10**(-10)
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
        self.Q = 100*np.ones( (self.N_States,len(self.actions) ))
        # Set goal to 0
        flatGoalState = self.flatten(self.goal[0],self.goal[1])
        self.Q[flatGoalState,:]=0
        return
    
    DP_SIMPLE=1

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

    # TODO: Remove
    def policy_evaluation(self,iterationBreaks=[],debug_state=-1):
        #self.initializePolicy()              
        #self.initRewardMatrix()   
        #self.initPolicyTransitionMatrix()
        debug = 0
        delta=0
        x_range = []
        #debug_state=11
        i=0
        isConverged = False
        #for i in range(self.iterations):
        while not isConverged and i<self.max_iterations:
            i+=1
            
            newValue_function = np.zeros( self.world.shape )
            for state in range(self.N_States):
                outerSum = 0
                if self.isObstacle(state) or self.isGoal(state):
                    continue;

                if debug_state>-1 and state==debug_state or debug==1:
                    print("********* Start policy_evaluation for state {} **************".format(state))
                # We apply the policy for a state for each action
                for action in range(len(self.policy[state])):
                    # action in a policy will have a percent, if zero it was removed from policy
                    # transitions already pruned the terminal states which shouldnt get values
                    if self.policy[state,action] == 0.0:
                        continue
                    innerSum=0
                    # For deterministic case this should only be 1 transition per action
                    for probability, nextState, reward_nextState in self.probability_transitions[state][action]:
                        # if(isTerminalState == True):
                        #     continue
                        if debug_state>-1 and state==debug_state or debug==1:
                            print("state {}, reward {}, action {}, prob: {} nextState: {}"
                                  .format(state,reward_nextState,action,probability, nextState) )
                            print("inner sum {}".format(innerSum))
                            print("valuefunction for next state from before {}".format(self.value_function.flat[nextState]))
                        innerSum = innerSum + probability*(reward_nextState + self.gamma*self.value_function.flat[nextState])
                        if debug_state>-1 and state==debug_state or debug==1:
                            print("New Inner Sum{}".format(innerSum))
                    
                    
                    if debug_state>-1 and state==debug_state or debug==1:
                        print("Policy {}".format(self.policy[state,action]))
                        print("State {}, Old Outer Sum {} for action: {}".format(state,outerSum,action))
                    outerSum = outerSum + self.policy[state,action]*innerSum
                    if debug_state>-1 and state==debug_state or debug==1:
                        print("State {}, New Outer Sum {} for action: {}".format(state,outerSum,action))
                if debug_state>-1 and state==debug_state or debug==1:
                    print("State {}, Final Q {}".format(state,outerSum))
                    print("********* End Policy Evaluation for state: {} **************".format(state))
                if(self.isGoal(state)):
                    outerSum = self.rewards_matrix.flat[state]
                newValue_function.flat[state]=outerSum
                deltaSingle = abs(self.value_function.flat[state]-newValue_function.flat[state])
                deltaMax = max(self.convergenceTolerance, deltaSingle)
                # if deltaMax<self.convergenceTolerance:
                #     isConverged=False

                # if self.isGoal(state)==False:
                #     newValue_function.flat[state]=outerSum
                # else:
                #     newValue_function.flat[state]=0
            delta = np.max(np.abs(newValue_function-self.value_function))
            x_range.append(delta)
            if debug_state>-1 or debug==1:
                print("State {}, Old Q(p) {}".format(debug_state,self.value_function.flat[debug_state]))
                
            self.value_function = newValue_function
            if debug_state>-1 or debug==1:
                print("State {}, NEW Q(p) {}".format(debug_state,self.value_function.flat[debug_state]))
                print("********* End Policy Evaluation for state: {} **************".format(debug_state))
            if delta < self.convergenceTolerance:
                print('Iterative policy evaluation algorithm converged! Iterations: {}'.format(i))
                isConverged=True

            if i in iterationBreaks:
              self.plot_matrix(self.value_function)
        self.iterations+=i
        print("Policy being returned, did {} iterations".format(self.iterations))
       
        return self.value_function ,x_range


    def policy_improvement(self,iterationBreaks=[],debug_state=-1):
        debug = 0
        debug_state=debug_state
        delta=0
        x_range = []
        policy_stable=True

        for state in range(self.N_States):
            if debug_state>-1 and state==debug_state or debug==1:
                print("**********Start Policy Improvement State {}".format(state))

            actionValues=np.zeros(len(self.policy[state]))
            
            for action in range(len(self.actions)):
                innerSum = 0
                for probability, nextState, reward_State in self.probability_transitions[state][action]:
                    # if(isTerminalState == True):
                    #     continue
                    if debug_state>-1 and state==debug_state or debug==1:
                        print("innerSum: {0}, prob: {1} nextState:{2} rew: {3}, isTerm: {4}".format(
                            innerSum,probability, nextState, reward_State) )
                        print("valuefunction for next state from before {}".format(self.value_function.flat[nextState]))

                    innerSum = innerSum + probability*(reward_State + self.gamma*self.value_function.flat[nextState])

                    if debug_state>-1 and state==debug_state or debug==1:
                        print("New Inner Sum{}".format(innerSum))
                
                actionValues[action] = actionValues[action] + innerSum
            
            
            # if(self.policy[state,maxValueIndex]<=0):
            #         # We found policy with better Q values
            #         policy_stable=False
            state_policy_stable = self.updatePolicy(state,actionValues,debug_state)

            if(state_policy_stable==False):
                policy_stable=False


            if debug_state>-1 and state==debug_state or debug==1:
                    print("**********END Policy Improvement for state: {}".format(self.policy[state]))
            
            # for actionindex in indicesOfMax[0]:
            #     if self.policy[state,actionindex]<=0:
            #         policy_stable = False
            # if policy_stable == False:
            #     for action in range(len(self.actions)):
            #         if np.any(indicesOfMax == action):
            #             self.policy[state,action]=1/len(indicesOfMax)
            #         else:
            #             self.policy[state,action]=0
        print("Policy being returned {}".format(policy_stable))
        return policy_stable

    def updatePolicy(self,state,actionValues,  debug_state  ):
        debug = 0
        policyStable = True
        if (self.isGoal(state) or self.isObstacle(state)):
            return policyStable 
        
        maxValueIndex = np.argmax(actionValues)
        # select all matching values and get indices
        indicesOfNewMax = np.where(actionValues==actionValues[maxValueIndex])[actionValues.ndim-1]
        oldMaxValueArg = np.argmax(self.policy[state,:])
        # select all matching values and get indices
        indicesOfOldMax = np.where(self.policy[state,:] == self.policy[state,oldMaxValueArg])[self.policy[state,:].ndim-1]
        
        if debug_state>-1 and state==debug_state or debug==1:
                print("**********Action State Values {}".format(actionValues))
        if debug_state>-1 and state==debug_state or debug==1:
                print("**********Current Policy  {}".format(self.policy[state]))


        x=np.sort(indicesOfOldMax)
        y=np.sort(indicesOfNewMax)
        if np.array_equiv(x, y) !=True: 
            policyStable = False
            self.policy[state,:]=np.zeros((len(self.actions)))
            for i in indicesOfNewMax:
                self.policy[state,i] = 1 / len(indicesOfNewMax)
        if debug_state == True and policyStable==True:
            print('Stable!!!')
        
        return policyStable

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
                if self.isObstacle(state1):
                    state1 = self.flatten(self.start[0],self.start[1])
                action1 = self.getActionFromStateDerivedQ(state1)
                self.Q[currentState,currentAction] = self.Q[currentState,currentAction] +self.alpha*(reward1 + self.gamma*self.Q[state1,action1] - self.Q[currentState,currentAction])
              
                currentState = state1
                currentAction = action1

        print("Completed run() on episodes")
        # policyStable=False
        # iterationsToShowGraph=[]
        # # for i in range(self.iterations):
        # while policyStable == False:
        #     self.policy_evaluation(iterationsToShowGraph,debug_state)
        #     policyStable = self.policy_improvement(iterationsToShowGraph, debug_state)
    


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
   

   
        

    def plot_grid(self):
        """Plot grid
        Args:
            env (Environment): grid world environment
        """

        data = self.rewards_matrix.copy()
        data = data.reshape((15,51))
        #data[env.e_x, env.e_y] = 10

        # create discrete colormap
        cmap = colors.ListedColormap(['grey', 'white', 'red'])
        bounds = [-51, -2, 0, 51]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, norm=norm)

        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, self.N_States, 1))
        ax.set_yticks(np.arange(-.5, self.N_States, 1))

        plt.show()            
    