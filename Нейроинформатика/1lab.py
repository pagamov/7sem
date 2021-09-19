import random
import numpy as np
import matplotlib.pyplot as plt

x1 = [[-4,-3.6],[-3.4,1.2],
      [0.7,-4.5],[4.3,2.2],
      [2.3,-4.4],[3.6,4.3]]
y1 = [0,1,0,0,0,1]

x2 = [[4.3,-3.1],[-2.5,3.9],
      [0.9,0],[1.1,3.1],
      [0.3,-3],[-0.5,-0.8],
      [4.6,1.2],[1.9,2.2]]
y2 = [0,1,0,0,1,1,0,0]
y22 =[0,1,1,1,0,1,0,1]

def draw(p,x1,y1):
    x = np.linspace(-5,5,50)
    y = np.linspace(-5,5,50)
    xt,yt = [],[]
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            if p([x[i],y[j]]) != p([x[i+1],y[j+1]]):
                xt.append(x[i])
                yt.append(y[j])
    plt.plot(xt, yt,'r')
    for k in range(len(x1)):
        i = x1[k]
        if y1[k] == 1:
            plt.scatter(i[0],i[1],c='b')
        else:
            plt.scatter(i[0],i[1],c='g')
    plt.grid()
    plt.legend()
    plt.show()

class Percerptron:
    def __init__(self, inp, outp, a=-5,b=5):
        self.inp = inp
        self.outp = outp
        self.w = [random.randint(-a,b) for x in range(inp)]
        # self.w0 = random.randint(-a,b)

    def __call__(self,data):
        return 1 if sum([self.w[i]*data[i] for i in range(len(data))]) >= 0 else 0

    def learn(self,data,res):
        allerr = 0
        for d in range(len(data)):
            s = sum([self.w[i]*data[d][i] for i in range(len(data[d]))])
            r = 1 if s >= 0 else 0
            error = res[d] - r
            allerr = allerr + abs(error)
            delta = (error) * 1 / (1 + 2.7 ** -s)
            # if r != res[d]:
            self.w = [self.w[i] + data[d][i] * delta for i in range(len(data[d]))]
        # print(allerr)

p = Percerptron(2,1)
for i in range(10):
    p.learn(x1,y1)
# exit()
# draw(p,x1,y1)

x1 = [[-2,-1],[1,2],
      [2,-1],[4,3.5],
      [4,-4],[-4,0]]
y1 = [0,0,0,1,1,1]

p = Percerptron(2,1)
p.learn(x1,y1)
# draw(p,x1,y1)

p = Percerptron(2,1)
p.learn(x2,y2)
# draw(p,x2,y2)
