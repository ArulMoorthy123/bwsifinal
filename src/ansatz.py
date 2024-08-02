import numpy as np
from qiskit import QuantumCircuit

def Ansatz(num_latent, num_trash, theta_list) -> QuantumCircuit:
    # if(not(len(theta_list) == num_trash + num_latent)):
    #     raise Exception("invalid theta_list length while creating Ansatz")

    ansatz_circuit = QuantumCircuit(num_latent + num_trash)

    # print(type(theta_list))

    # ry gates on all qubits
    n = num_latent + num_trash
    ry_circuit = QuantumCircuit(num_latent + num_trash)
    for i in range(len(theta_list)):
        ry_circuit.ry(theta_list[i], i)

    trash_circuit = QuantumCircuit(num_latent + num_trash)
    for i in range(num_trash):
        for j in range(i + 1, num_trash):
            trash_circuit.cz(i + num_latent, j + num_latent)
    
    # different trash-latent pairing
    ansatz_circuit.compose(ry_circuit, inplace=True)
    ansatz_circuit.compose(trash_circuit, inplace=True)
    for i in range(num_latent):
        ansatz_circuit.cz(i % num_trash + num_latent, i)

    for rep in range(num_trash - 1):
        ansatz_circuit.barrier()
        ansatz_circuit.compose(ry_circuit, inplace=True)
        ansatz_circuit.compose(trash_circuit, inplace=True)
        for i in range(num_latent):
            if(i == rep):
                ansatz_circuit.cz((i + 1) % num_trash + num_latent, i)
            elif(i == rep + 1):
                ansatz_circuit.cz((i - 1) % num_trash + num_latent, i)
            else:
                ansatz_circuit.cz(i % num_trash + num_latent, i)

    ansatz_circuit.barrier()
    for i in range(num_latent, num_latent + num_trash):
        ansatz_circuit.ry(theta_list[i], i)
    
    # print(ansatz_circuit.draw())

    return ansatz_circuit



    
