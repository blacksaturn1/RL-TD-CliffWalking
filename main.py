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
        sarsaAlgo = Sarsa(W,max_episodes=20000,gamma=1.0,alpha=.5,epsilon=0.1,reduceEpsilon=False)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa-Path")
        sarsa_rewardsPerEpisode = sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        fig.savefig("Sarsa-Path.png")

        qAlgo = QLearning(W,max_episodes=20000,gamma=1.0,alpha=.5,epsilon=0.1,reduceEpsilon=False)
        draw=Helper()
        fig, ax = draw.plot_matrix(qAlgo.rewards_matrix,printValues=True,fontSize=5,title="Q-Learning-Path")
        QLearning_rewardsPerEpisode=qAlgo.run()
        draw.plot_max_quivers(ax,qAlgo.getPolicyQuivers())
        fig.savefig("Q-Learning-Path.png")

        x = np.arange(1,len(sarsa_rewardsPerEpisode)+1)
        
        plt.plot(x,sarsa_rewardsPerEpisode,label = "Sarsa")
        plt.plot(x,QLearning_rewardsPerEpisode,label = "Q-Learning")
        plt.legend()
        plt.savefig("Sarsa and Q-Learning-Sum of Rewards by Episode.png")
        plt.show()
        
        
   
        sarsaAlgo = Sarsa(W,max_episodes=20000,gamma=1.0,alpha=.5,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa Reduced Epsilon-Path")
        sarsa_reducedEpsilon_rewardsPerEpisode = sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        fig.savefig("Sarsa-Reduced Epsilon-Path.png")

        qAlgo = QLearning(W,max_episodes=20000,gamma=1.0,alpha=.5,epsilon=0.1,reduceEpsilon=True)
        draw=Helper()
        fig, ax = draw.plot_matrix(qAlgo.rewards_matrix,printValues=True,fontSize=5,title="Q-Learning Reduced Epsilon-Path")
        QLearning_rewardsPerEpisode=qAlgo.run()
        draw.plot_max_quivers(ax,qAlgo.getPolicyQuivers())
        fig.savefig("Q-Learning-Reduced Epsilon-Path.png")

    input("Press Enter to continue...")
    


if __name__ == "__main__":
    main()