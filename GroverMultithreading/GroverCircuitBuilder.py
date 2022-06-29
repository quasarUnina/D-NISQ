from qiskit import *
from qiskit.quantum_info.operators import Operator
from math import *
import numpy as np


def create_database(dimension, winner_item, filler_item, winner_index):

    database = [filler_item for _ in range(dimension)]
    database[winner_index] = winner_item

    return database


def create_sub_databases(main_database, number_of_subproblems, sub_dimension, winner_item):

    sub_databases = [[] for _ in range(number_of_subproblems)]
    k = 0
    for i in range(number_of_subproblems):
        for j in range(int(k), int(k + sub_dimension)):
            sub_databases[i].append(main_database[j])
        k += sub_dimension

    winner_location = []
    # winner_location = [[which sub_database has the winner item in it],[winner index in that sub_database]]
    for i in range(number_of_subproblems):
        if winner_item in sub_databases[i]:
            winner_location.append(i)
    for i in range(sub_dimension):
        if sub_databases[winner_location[0]][i] == winner_item:
            winner_location.append(i)

    return sub_databases, winner_location


def create_sub_basis(number_of_subproblems, sub_dimension, qubits_per_subproblem):

    sub_basis = [[] for _ in range(number_of_subproblems)]
    for i in range(number_of_subproblems):
        for j in range(sub_dimension):
            a = bin(j)[2:]
            l = len(a)
            b = str(0) * (qubits_per_subproblem - l) + a
            sub_basis[i].append(b)

    return sub_basis


def scalar_product(a, b):  # where a and b are two strings of a basis list

    if a == b:
        return 1
    else:
        return 0


def create_sub_oracles(number_of_subproblems, sub_dimension, winner_location):

    #global sub_oracles_matrix

    sub_oracles_matrix = [np.identity(sub_dimension) for _ in range(number_of_subproblems)]
    sub_oracles_matrix[winner_location[0]][winner_location[1], winner_location[1]] = -1

    sub_oracles = [Operator(sub_oracles_matrix[i]) for i in range(number_of_subproblems)]

    return sub_oracles


def create_sub_diffusers(number_of_subproblems, sub_dimension, sub_basis):

    sub_diffuser_matrix = np.empty((sub_dimension, sub_dimension))

    for k in range(number_of_subproblems):
        for i in range(sub_dimension):
            for j in range(sub_dimension):
                sub_diffuser_matrix[i, j] = (2 * scalar_product(sub_basis[k][i], sub_basis[k][0]) * scalar_product(
                    sub_basis[k][0], sub_basis[k][j])) - scalar_product(sub_basis[k][i], sub_basis[k][j])

    sub_diffusers = [Operator(sub_diffuser_matrix) for _ in range(number_of_subproblems)]

    return sub_diffusers


def grover_circuit_builder(sub_dimension, qubits_per_subproblem, sub_oracle, sub_diffuser, manual_it):

    all_qubits_list = [i for i in range(qubits_per_subproblem)]

    qr = QuantumRegister(qubits_per_subproblem, 'q')
    cr = ClassicalRegister(qubits_per_subproblem, 'c')
    grover_circuit = QuantumCircuit(qr, cr)

    # Hadamard layer
    for i in range(qubits_per_subproblem):
        grover_circuit.h(qr[i])

    # Set number of Grover iterations (oracle + diffuser) in the circuit
    if manual_it:
        grover_iterations = manual_it
    elif qubits_per_subproblem == 2:
        grover_iterations = 1
    else:
        grover_iterations = round(sqrt(sub_dimension))

    # Grover iteration layers
    for i in range(grover_iterations):
        # Oracle layer
        grover_circuit.unitary(sub_oracle, all_qubits_list, label="Oracle")
        # Hadamard layer
        for j in range(qubits_per_subproblem):
            grover_circuit.h(qr[j])
        # Diffuser layer
        grover_circuit.unitary(sub_diffuser, all_qubits_list, label="Diffuser")
        # Hadamard layer
        for j in range(qubits_per_subproblem):
            grover_circuit.h(qr[j])

    # Measurement layer
    grover_circuit.measure(qr, cr)

    return grover_circuit


def verify_success(number_of_subproblems, sub_dimension, qubits_per_subproblem, sub_counts, expected_index):

    winner_location = [None, None]

    max_counts_list = []
    for i in range(number_of_subproblems):
        values_list = list(sub_counts[i].values())
        max_count = max(values_list)
        max_counts_list.append(max_count)

    very_max_count = max(max_counts_list)
    winner_location[0] = max_counts_list.index(very_max_count)

    if number_of_subproblems == 1:
        for j in range(sub_dimension):
            a = bin(j)[2:]
            l = len(a)
            b = str(0) * (qubits_per_subproblem - l) + a
            if b in sub_counts[0].keys():
                if sub_counts[0][b] == very_max_count:
                    winner_location[1] = j
    else:
        for i in range(number_of_subproblems):
            for j in range(sub_dimension):
                a = bin(j)[2:]
                l = len(a)
                b = str(0) * (qubits_per_subproblem - l) + a
                if b in sub_counts[i].keys():
                    if sub_counts[i][b] == very_max_count:
                        winner_location[1] = j

    main_index = (winner_location[0] * sub_dimension) + winner_location[1]

    if main_index == expected_index:
        success = True
    else:
        success = False

    information_list = []
    information_list.append(f"\nWinner item has been found in sub-problem number {winner_location[0] + 1}.")
    information_list.append(f"Index in main database: {main_index}.")
    information_list.append(f"Expected index: {expected_index}.")
    if success:
        information_list.append("\nGrover's search has succeeded.")
    else:
        information_list.append("\nGrover's search has failed.")
    success_statement = "\n".join(information_list)

    return success, success_statement





