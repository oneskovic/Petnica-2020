import matplotlib.pyplot as plt

class Plotter():
    def __init__(self):
        pass
    def __plot_organisms(self,organisms,cmap):
        x_positions = []
        y_positions = []
        hps = []
        for i in range(len(organisms)):
            x_positions.append(organisms[i][0])
            y_positions.append(organisms[i][1])
            hps.append(organisms[i][2])
        plt.scatter(x_positions, y_positions,c = hps,cmap=cmap,vmin=0,vmax=20)
        
    def plot_state(self,state,directory):
        i = 0
        red_organisms = []
        blue_organisms = []
        green_organisms = []
        while i < len(state):
            x = state[i]
            y = state[i+1]
            hp = state[i+2]
            type = state[i+4]
            if type == 0:
                green_organisms.append([x,y,hp])
            elif type == 1:
                blue_organisms.append([x,y,hp])
            else:
                red_organisms.append([x,y,hp])
            i += 6
        plt.cla()
        plt.xlim(0,10)
        plt.ylim(0,10)
        self.__plot_organisms(red_organisms,'Reds')
        self.__plot_organisms(blue_organisms,'Blues')
        self.__plot_organisms(green_organisms,'Greens')
        plt.savefig(directory)
        plt.cla()