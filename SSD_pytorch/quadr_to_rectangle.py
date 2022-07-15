import sympy
from sympy import *
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quadr2rectangle(img_input, t=1):
    img_output = []
    img_output = [[0] * 3 for _ in range(4)]

    x1 = img_input[0][0]
    y1 = img_input[0][1]
    z1 = img_input[0][2]

    x2 = img_input[1][0]
    y2 = img_input[1][1]
    z2 = img_input[1][2]

    x3 = img_input[2][0]
    y3 = img_input[2][1]
    z3 = img_input[2][2]

    x4 = img_input[3][0]
    y4 = img_input[3][1]
    z4 = img_input[3][2]

    a1 = x1 ** 2 + y1 ** 2 + z1 ** 2
    a2 = x1 * x4 + y1 * y4 + z1 * z4 - x1 * x2 - y1 * y2 - z1 * z2
    a3 = -1 * (x1 * x4 + y1 * y4 + z1 * z4)
    a4 = -1 * (x2 * x4 + y2 * y4 + z2 * z4)
    a5 = x2 * x4 + y2 * y4 + z2 * z4

    b1 = x3 ** 2 + y3 ** 2 + z3 ** 2
    b2 = x2 * x3 + y2 * y3 + z2 * z3 - x3 * x4 - y3 * y4 - z3 * z4
    b3 = x3 * x4 + y3 * y4 + z3 * z4 - 2 * (x3 ** 2 + y3 ** 2 + z3 ** 2)
    b4 = -1 * (x2 * x4 + y2 * y4 + z2 * z4)
    b5 = x3 * x4 + y3 * y4 + z3 * z4 + x2 * x4 + y2 * y4 + z2 * z4 - x2 * x3 - y2 * y3 - z2 * z3
    b6 = -1 * (x3 * x4 + y3 * y4 + z3 * z4) + x3 ** 2 + y3 ** 2 + z3 ** 2

    m = Symbol('m')
    n = Symbol('n')
    solved_value = solve([a1 * m ** 2 + a2 * m * n + a3 * m + a4 * n ** 2 + a5 * n,
                          b1 * m ** 2 + b2 * m * n + b3 * m + b4 * n ** 2 + b5 * n + b6], [m, n])

    npdata = np.array(solved_value)
    # npdata = npdata.astype(np.float64)

    theta = []
    for i, j in enumerate(npdata):
        for k in j:
            if not (isinstance(k, sympy.core.mul.Mul) or
                    isinstance(k, sympy.core.add.Add)):
                result = k.evalf()
                if result > 0:
                    theta.append(result)
                    print(i, theta)

    img_output[0][0] = round(t * theta[0] * x1, 3)
    img_output[0][1] = round(t * theta[0] * y1, 3)
    img_output[0][2] = round(t * theta[0] * z1, 3)

    img_output[1][0] = round(t * theta[1] * x2, 3)
    img_output[1][1] = round(t * theta[1] * y2, 3)
    img_output[1][2] = round(t * theta[1] * z2, 3)

    img_output[2][0] = round(t * (1 - theta[0]) * x3, 3)
    img_output[2][1] = round(t * (1 - theta[0]) * y3, 3)
    img_output[2][2] = round(t * (1 - theta[0]) * z3, 3)

    img_output[3][0] = round(t * (1 - theta[1]) * x4, 3)
    img_output[3][1] = round(t * (1 - theta[1]) * y4, 3)
    img_output[3][2] = round(t * (1 - theta[1]) * z4, 3)

    return img_output


img_input = [[-2, 1, 1], [-1, 2, 1], [1, 2, 1], [2, 1, 1]]
img_output = quadr2rectangle(img_input, 1)

for i in range(len(img_output)):
    for j in range(len(img_output[0])):
        img_output[i][j] = round(img_output[i][j], 3)

print("img_output=\n",img_output)
xq,yq,zq,xr,yr,zr= [list() for x in range(6)]
for i in range(4):
    xq.append(img_input[i][0])
    xr.append(img_output[i][0])
for i in range(4):
    yq.append(img_input[i][1])
    yr.append(img_output[i][1])
for i in range(4):
    zq.append(img_input[i][2])
    zr.append(img_output[i][2])

xq.append(img_input[0][0])
xr.append(img_output[0][0])
yq.append(img_input[0][1])
yr.append(img_output[0][1])
zq.append(img_input[0][2])
zr.append(img_output[0][2])

fig = plt.figure()
ax = fig.add_axes((0.1,0.1,0.8,0.8), projection='3d')
ax.plot(xq, yq, zq, c='red', marker='o')
ax.plot(xr, yr, zr, c='green', marker='o')

plt.show()
