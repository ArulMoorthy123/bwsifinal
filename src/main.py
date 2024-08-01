import encodeData as ED
from ansatz import Ansatz
import autoEncoder as AE
import costfunc as CF
from particleSwarmOptimizer import Swarm, Particle

from autoEncoder import AutoEncoder


import numpy as np
import random

CATEGORIES = 7

def main():
    (normal_data, fraud_data)  = ED.EncodeData(CATEGORIES, "../data/test_data.csv")
    # theta_list = [np.random(0, np.pi)] * CATEGORIES
    # Ansatz(normal_data.num_qubits())
    particleNum = 64

    swarm = Swarm([Particle(CATEGORIES, [0, np.pi], random.uniform(0.9, 1.1)) for i in range(particleNum)], CATEGORIES, [0, np.pi])
        
    d = 0
    
    x = 0

    while not d and x < 200:
        print(x)
        d = swarm.stepAlgorithm(normal_data, fraud_data)
        x += 1

    print(d)

    


if __name__ == "__main__":
    main()

