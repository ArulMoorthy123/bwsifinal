import encodeData as ED
from ansatz import Ansatz
import autoEncoder as AE
import costfunc as CF
from particleSwarmOptimizer import Swarm, Particle

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator, Aer
import numpy as np
import random

CATEGORIES = 7

def main():
    (normal_data, fraud_data)  = ED.EncodeData(CATEGORIES, "../data/test_data.csv")

    print("classical data embedded into qubits")

    particleNum = 10

    swarm = Swarm([Particle(CATEGORIES, [0, np.pi], random.uniform(0.9, 1.1)) for i in range(particleNum)], CATEGORIES, [0, np.pi])
        
    d = 0
    
    x = 0

    while not d and x < 200:
        print("iteration #" + str(x))
        d = swarm.stepAlgorithm(normal_data, fraud_data)
        x += 1

    print(d)


if __name__ == "__main__":
    main()

