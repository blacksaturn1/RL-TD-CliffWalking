
import matplotlib.pyplot as plt
import numpy as np

class Helper:
    def __main__(self):
        return
    
    def getStateValue(self, Q):
        value_function = np.zeros( len(Q) )
        for x in range(len(Q)):
            value_function[x]=sum(Q[x,:])
        return value_function


    
    def plot_max_quivers(self,axis,policyQuivers):
        x_pos, y_pos, x_direct, y_direct=policyQuivers
        axis.quiver(x_pos,y_pos,x_direct,y_direct,angles='xy', scale_units='xy', scale=2)
        plt.show()

    def plot_max_quivers_test(self,axis):
        x_pos=[0, 1, 2, 3]
        y_pos=[0, 0, 0, 0]
        x_direct=[0, 1, 0, -1]
        y_direct=[1, 0, -1, 0]
        for i in range(4):
            x_pos.append(i)
            y_pos.append(0)
            x_direct.append(1)
            y_direct.append(0)
        axis.quiver(x_pos,y_pos,x_direct,y_direct,angles='xy', scale_units='xy', scale=2)
        plt.show()

    def plot_matrix(self, value_function,fig=None, ax=None,printValues=True,fontSize=5,title="Grid Values"):
        if ax==None:
            fig, ax = plt.subplots(figsize=(10,10))
           

        im = ax.imshow(value_function)
        ax.set_title(title)
        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        if printValues:
            # Loop over data dimensions and create text annotations.
            for i in range(value_function.shape[0]):
                for j in range(value_function.shape[1]):
                    text = ax.text(j, i, round(value_function[i, j],1),
                                ha="center", va="center", fontsize=fontSize, color="w")
        #ax.grid()
        #fig.savefig("test.png")
        #fig.tight_layout()
        
        # x_pos = 0
        # y_pos = 0
        # x_direct = 1
        # y_direct = 1

        # ax.quiver(x_pos,y_pos,x_direct,y_direct,angles='xy', scale_units='xy', scale=3)
        return fig,ax