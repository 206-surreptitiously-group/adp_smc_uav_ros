import numpy as np
import matplotlib.pyplot as plt


# dx = 2x + u
if __name__ == '__main__':
    pos_zone = np.atleast_2d([[-5, 5], [-5, 5], [0, 3]])

    x = np.random.uniform(low=pos_zone[0][0], high=pos_zone[0][1], size=2)
    y = np.random.uniform(low=pos_zone[1][0], high=pos_zone[1][1], size=2)
    z = np.random.uniform(low=pos_zone[2][0], high=pos_zone[2][1], size=2)
    print(x)
    print(y)
    print(z)
    print(np.vstack((x,y,z)))
