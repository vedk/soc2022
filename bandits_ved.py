import numpy as np

MEAN = 1
STD = 1
NARMS = 5
EPSILON = 0.1
ITER = 3000

true_dist = np.round(
    np.absolute(np.random.normal(loc=MEAN, scale=STD, size=NARMS)), decimals=2
)
print("True Means:", true_dist)

dist = np.zeros(shape=NARMS, dtype=float)

for n in range(1, ITER // 3):
    i = np.random.choice(NARMS)
    reward = abs(np.random.normal(loc=true_dist[i], scale=STD))
    dist[i] = dist[i] + ((reward - dist[i]) / n)

for n in range(ITER//3, ITER):
    if np.random.random_sample() < EPSILON:
        i = np.random.choice(NARMS)
    else:
        i = np.argmax(dist)

    reward = np.round(abs(np.random.normal(loc=true_dist[i], scale=STD)), decimals=2)
    dist[i] = dist[i] + ((reward - dist[i]) / n)

dist = np.round(dist, decimals=2)
print("Estd Means:", dist)
print("Rel Errors:", np.round(np.absolute(true_dist - dist) / true_dist, decimals=2))
