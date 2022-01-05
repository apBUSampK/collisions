import numpy as np
import matplotlib.pyplot as plt

dt = .001
g = np.array((0., -9.8))


def calc(alpha, h, t):
    tau = np.arange(0, t, dt)
    cycles = int(t / dt)
    k = np.tan(alpha)
    floor = np.array((np.cos(alpha), np.sin(alpha)))
    print(floor)
    coord = np.array((0., h))
    v = np.array((0., 0.))
    # containers for data
    x = np.empty(cycles)
    y = np.empty(cycles)
    vx = np.empty(cycles)
    vy = np.empty(cycles)
    x[0] = 0.
    y[0] = h
    vx[0] = 0.
    vy[0] = 0.
    for i in range(1, cycles):
        buff = coord + v * dt
        if buff[1] > k * buff[0]:
            # next position is above the plane
            coord = buff
            v += g * dt
        else:
            # next position is below the plane
            # find distance vector to the plane
            dist = coord - floor * np.dot(coord, floor)
            # find time to intersection
            tcycle = np.dot(dist, dist) / abs(np.dot(dist, v))
            coord += v * tcycle
            v += g * tcycle
            # reflect the speed
            v -= 2 * dist * np.dot(dist, v) / np.dot(dist, dist)
            coord += v * (dt - tcycle)
            v += g * (dt - tcycle)
        x[i] = coord[0]
        y[i] = coord[1]
        vx[i] = v[0]
        vy[i] = v[1]
    return x, y, vx, vy, tau


def plotter(x, y, vx, vy, tau, k):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(x, y)
    ax.plot((0, x[-1]), (0, k * x[-1]), color='k')
    ax = fig.add_subplot(312)
    ax.plot(tau, x)
    ax = fig.add_subplot(313)
    ax.plot(tau, y)
    plt.show()


def main():
    alpha = float(input("Enter the angle (counter-clockwise) in degrees, ranging from -90 to 90: "))
    alpha = alpha * np.pi / 180
    h = float(input("Enter the height: "))
    t = float(input("Enter simulation time: "))
    plotter(*calc(alpha, h, t), np.tan(alpha))


if __name__ == "__main__":
    main()
