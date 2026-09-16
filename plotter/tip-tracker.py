import matplotlib.pyplot as plt
import numpy as np

times = [12000 * n for n in range(11)]
print(times)
speeds = [
    0,
    0.0111077,
    0.0718651,
    0.340726,
    0.403716,
    0.645283,
    0.780202,
    0.57256,
    1.04459,
    1.17548,
    0.462115,
]


def pos(t):
    accumulator = 0
    dt = 12000
    if t < 0:
        return accumulator
    for i in range(10):
        start = i * dt
        end = (i + 1) * dt
        avg_speed = (speeds[i] + speeds[i + 1]) / 2
        if start <= t < end:
            return (t - start) * avg_speed + accumulator
        else:
            accumulator += (end - start) * avg_speed
    return accumulator


positions = [pos(time) for time in times]
microseconds = list(range(121))
snapshots = list(range(60000))
print(max(positions))

vels = np.interp(snapshots, positions, speeds)

fig, ax = plt.subplots()
ax.plot(snapshots, vels)
ax.set_xlim(0, 20000)
for i in range(10):
    ax.axhline(i * 0.1)
for i in [0, 0.8, 1.75, 2.65, 7.3, 10, 12.6, 17.4]:
    ax.axvline(i * 1000)
fig.savefig("gloop")
