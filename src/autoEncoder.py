import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from ansatz import Ansatz
from qiskit_aer import AerSimulator, Aer
from qiskit.compiler import assemble
from qiskit.circuit.library import RealAmplitudes

def AutoEncoder(data, fraud, num_categories, theta_list) -> tuple:
    num_trash = 2
    num_latent = num_categories - num_trash
    n = data.num_qubits + num_trash + 1

    # print("total number of qubits: " + str(n))

    autoencoder_sz = num_trash * 2 + num_latent + 1
    auxiliary_qubit = n - 1

    if(autoencoder_sz != 10):   # for safety
        raise Exception("autoencoder_sz incorrect, got: " + str(autoencoder_sz))

    encoder_circuit = QuantumCircuit(n)
    encoder_circuit.compose(Ansatz(num_latent, num_trash, theta_list), range(n - autoencoder_sz, n - autoencoder_sz + num_trash + num_latent), inplace=True)

    swap_circuit = QuantumCircuit(n)
    swap_circuit.h(auxiliary_qubit)
    for i in range(n - 1 - 2 * num_trash, n - 1 - num_trash):
        swap_circuit.cswap(auxiliary_qubit, i, i + num_trash)
    swap_circuit.h(auxiliary_qubit)

    quantum_register = QuantumRegister(n)
    classical_register = ClassicalRegister(1)
    qc = QuantumCircuit(quantum_register, classical_register)
    qc.compose(data, range(data.num_qubits), inplace=True)
    qc.barrier()
    qc.compose(encoder_circuit, inplace=True)
    qc.barrier()
    qc.compose(swap_circuit, inplace=True)
    qc.barrier()
    qc.save_density_matrix(qubits=[auxiliary_qubit])

    # print(qc.draw())

    simulator = AerSimulator()
    circ = transpile(qc, backend=simulator)
    job = simulator.run(circ)
    state = job.result().data()['density_matrix']

    print("prediction made")

    return(float(state[0][0]), fraud)
