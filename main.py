import numpy as np 
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
    
    showGraphs=True
    
    if not isDebugPolicy:
        sarsaAlgo = Sarsa(W,max_episodes=5000,gamma=1,alpha=.5,epsilon=0.1)
        draw=Helper()
        fig, ax = draw.plot_matrix(sarsaAlgo.rewards_matrix,printValues=True,fontSize=5,title="Sarsa-MaxQ Path")
        sarsaAlgo.run()
        draw.plot_max_quivers(ax,sarsaAlgo.getPolicyQuivers())
        title = "Sarsa"
        fig.savefig(title+"-Max Q Path.png")
    

   
    input("Press Enter to continue...")
    


if __name__ == "__main__":
    main()