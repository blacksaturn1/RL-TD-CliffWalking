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
        sarsaAlgo = Sarsa(W,max_episodes=10000,gamma=1,alpha=.1,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa-Path")
        sarsa_rewardsPerEpisode = sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        title = "Sarsa"
        fig.savefig(title+"-Path.png")

        x = np.arange(1,len(sarsa_rewardsPerEpisode)+1)
        plt.plot(x,sarsa_rewardsPerEpisode,label = "Sarsa")
        plt.legend()
        plt.show()
        
        qAlgo = QLearning(W,max_episodes=1000,gamma=1,alpha=.1,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(qAlgo.rewards_matrix,printValues=True,fontSize=5,title="Q-Learning-Path")
        qAlgo.run()
        draw.plot_max_quivers(ax,qAlgo.getPolicyQuivers())
        title = "Q-Learning"
        fig.savefig(title+"-Path.png")

   
    input("Press Enter to continue...")
    


if __name__ == "__main__":
    main()