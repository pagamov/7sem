import matplotlib.pyplot as plt
data = []
x = []
y = []
with open("points.txt") as f:
    for line in f:
        data = [float(x) for x in line.split(" ")]
        x.append(data[0])
        y.append(data[1])
plt.plot(x, y,'ro')
plt.show()