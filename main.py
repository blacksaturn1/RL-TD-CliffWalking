# from PolicyIteration import PolicyIteration
# from GeneralPolicyIteration import GeneralPolicyIteration as gp
# from ValueIteration import ValueIteration
import numpy as np 
import time
import math
import matplotlib.pyplot as plt
from helper import Helper 
from SarsaAlgorithm import Sarsa

def main():
    W = [
            [0,0,0,0 ,0,0,0,0 ,0,0,0,0,],
            [0,0,0,0, 0,0,0,0, 0,0,0,0,],
            [0,0,0,0, 0,0,0,0, 0,0,0,0,],
            [0,1,1,1, 1,1,1,1, 1,1,1,0,]]
    
    
    isDebugPolicy=False
    if isDebugPolicy:
        W = [
                [0,0,0,0,],
                [0,0,0,0,],
                [0,0,0,0,],
                [0,0,0,0,],
                
            ]
        W = [
            [1,1,1,1,1,1,],
            [1,0,0,1,0,1,],
            [1,0,0,0,0,1,],
            [1,0,0,1,1,1,],
            [1,1,1,1,1,1,],
        ]
  
    # W = [
    #         [0,0],
    #         [0,0],
    #     ]
    
    
    
    # Toy problems
    #dp=PolicyIteration(W,iterations=3,gamma=1,goal=(0,0),rewards=[0,-1,-50])
    #dp=PolicyIteration(W,iterations=10,gamma=1,goal=(0,0),rewards=[0,-1,-50],actions=(0,1,2,3,4,5,6,7))
    
    showGraphs=True
    
    if not isDebugPolicy:
        # valueIteration=ValueIteration(W,gamma=0.95,goal=(0,0),rewards=[100,-1,-50],actions=(0,1,2,3), prob=[.1,.8,.1])
        # valueIteration=ValueIteration(W,gamma=0.95,goal=goalState,rewards=[100,-1,-50],actions=(0,1,2,3,4,5,6,7))
        sarsaAlgo = Sarsa(W,max_episodes=2000,gamma=1,alpha=.1,epsilon=0.1)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa-MaxQ Path")
        sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        title = "Sarsa"
        fig.savefig(title+"-Max Q Path.png")
    #     def RunSimulation(algo, title):
    #         algo.run()
    #         if showGraphs == True:
    #             draw=Helper()
    #             fig, ax = draw.plot_matrix(algo.value_function,printValues=True,fontSize=5,title=title+"-Value Function")
    #             fig.savefig(title+"-Value Function.png")        
    #             fig, ax = draw.plot_matrix(algo.rewards_matrix,printValues=False,fontSize=5,title=title+"-Policy")
    #             draw.plot_max_quivers(ax,algo.getPolicyQuivers())
    #             fig.savefig(title+"-Policy.png")
    #             return algo.iterations
        
        
    #     #print("Iterations: ",iterations_vD,iterations_vS,iterations_pD,iterations_pS,iterations_gpD,iterations_gpS)
       
    # else:
    #     # policyIteration=PolicyIteration(W,iterations=len(W[0]),gamma=1,goal=(3,3),rewards=[0,-1,-1],actions=(0,1,2,3))
    #     policyIteration=PolicyIteration(W,iterations=len(W[0]),gamma=1,goal=(2,4),rewards=[0,-1,-1],actions=(0,1,2,3,4,5,6,7))
    #     policyIteration.run(-1)
    #     if showGraphs == True:
    #         draw=Helper()
    #         fig, ax = draw.plot_matrix(policyIteration.rewards_matrix,printValues=True,fontSize=5,title="Policy Iteration-Rewards Matrix")
            
    #         fig, ax = draw.plot_matrix(policyIteration.value_function,printValues=True,fontSize=5,title="Policy Iteration-Value Function")
            
    #         fig, ax = draw.plot_matrix(policyIteration.rewards_matrix,printValues=False,fontSize=5,title="Policy Iteration-Policy")
    #         draw.plot_max_quivers(ax,policyIteration.getPolicyQuivers())
    #         fig.savefig("Policy Iteration-Policy.png")
        

   
    input("Press Enter to continue...")
    


if __name__ == "__main__":
    main()