import numpy as np 
import matplotlib.pyplot as plt
from helper import Helper 
from SarsaAlgorithm import Sarsa
from QLearningAlgorithm import QLearning

def main():
    W = [
            [0,0,0,0 ,0,0,0,0 ,0,0,0,0,],
            [0,0,0,0, 0,0,0,0, 0,0,0,0,],
            [0,0,0,0, 0,0,0,0, 0,0,0,0,],
            [0,1,1,1, 1,1,1,1, 1,1,1,0,]]
    
    isDebugPolicy=False
    
    showGraphs=True
    
    if not isDebugPolicy:
        sarsaAlgo = Sarsa(W,max_episodes=1000,gamma=1,alpha=.50,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa-Path")
        sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        title = "Sarsa"
        fig.savefig(title+"-Path.png")
    
        qAlgo = QLearning(W,max_episodes=1000,gamma=1,alpha=.50,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Q-Learning-Path")
        sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        title = "Q-Learning"
        fig.savefig(title+"-Path.png")

   
    input("Press Enter to continue...")
    


if __name__ == "__main__":
    main()