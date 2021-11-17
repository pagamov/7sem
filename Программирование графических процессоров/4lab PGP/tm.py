from random import randint

f = open('data', 'w')
n = int(input())
f.write(str(n) + '\n')

for i in range(n):
    for j in range(n):
        f.write(str(randint(1,256)) + ' ')
    f.write('\n')
f.close()