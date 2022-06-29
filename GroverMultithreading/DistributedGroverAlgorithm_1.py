from threading import *
from concurrent.futures import *
from datetime import datetime
from statistics import mean

from GroverCircuitBuilder import *
from QuantumComputing import *


def distributed_grover_search(number_of_subproblems, sub_grover_circuits, simulation, noise, shots, backend,
                              mock_backend_list, provider, thread_pool=False, max_workers=32):

    result_list = [None for _ in range(number_of_subproblems)]

    if thread_pool:

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(quantum_circuit_executer, i, sub_grover_circuits[i], simulation, noise, shots,
                                       backend, mock_backend_list, result_list, thread_pool, provider)
                       for i in range(number_of_subproblems)]
            result_list = [future.result() for future in futures]

        print("\nAll threads are done.")
        return result_list

    else:

        thread_list = [Thread(target=quantum_circuit_executer,
                              args=(i, sub_grover_circuits[i], simulation, noise, shots, backend, mock_backend_list,
                                    result_list, thread_pool, provider)) for i in range(number_of_subproblems)]

        for i in range(number_of_subproblems):
            thread_list[i].start()
            print(f"Thread {i + 1} started\n")
            thread_list[i].join()

        #for i in range(number_of_subproblems):
            #thread_list[i].join()

        print("\nAll threads are done.")

        return result_list


def ask_for_backends(number_of_subproblems):

    backend_list = [input(f"\nAssign backend to sub-problem number {i + 1}:") for i in range(number_of_subproblems)]

    information_list = [f"Sub-problem {i + 1}: {backend_list[i]}" for i in range(number_of_subproblems)]
    information_list.insert(0, "\nQUANTUM COMPUTER:")
    backend_information = "\n".join(information_list)

    return backend_list, backend_information


def get_counts(results, number_of_subproblems, thread_pool=False):

    sub_counts = [results[i].get_counts() for i in range(number_of_subproblems)]

    if thread_pool:
        information_list = [f"Sub-problem {i + 1}: {sub_counts[i]}" for i in range(number_of_subproblems)]
    else:
        information_list = [f"Thread {i + 1}: {sub_counts[i]}" for i in range(number_of_subproblems)]
    information_list.insert(0, "\nCOUNTS:")
    counts_information = "\n".join(information_list)

    return sub_counts, counts_information


def get_times(results, number_of_subproblems, thread_pool=False):

    execution_times = [results[i].time_taken for i in range(number_of_subproblems)]
    max_time = max(execution_times)

    if thread_pool:
        information_list = [f"Sub-problem {i + 1}: {execution_times[i]} s" for i in range(number_of_subproblems)]
        information_list.append(
            f"\nLongest execution time: {max_time} s (in sub-problem number {execution_times.index(max_time) + 1})")
    else:
        information_list = [f"Thread {i + 1}: {execution_times[i]} s" for i in range(number_of_subproblems)]
        information_list.append(
            f"\nLongest execution time: {max_time} s (in thread number {execution_times.index(max_time) + 1})")
    information_list.insert(0, "\nEXECUTION TIMES:")

    times_information = "\n".join(information_list)

    return execution_times, max_time, times_information


def current_situation(run_so_far, max_times_so_far, successes_so_far):

    information_list = []
    information_list.append("------------------------------------------------------------")
    information_list.append(f"After {run_so_far} runs:")
    information_list.append(f"Mean execution time: {mean(max_times_so_far)} s")
    information_list.append(f"Successes: {successes_so_far}")
    information_list.append(f"Success rate: {successes_so_far * 100 / run_so_far} %.")
    information_list.append("------------------------------------------------------------")

    current_sit_information = "\n".join(information_list)
    return current_sit_information


def report_opening(dateandtime, number_of_runs, dimension, number_of_subproblems, sub_dimension, qubits_per_subproblem,
                   manual_iter, shots, simulation, noise, diff_backend, backend):

    information_list = []
    information_list.append("\n___________________________________________________________")
    information_list.append("\nDISTRIBUTED GROVER'S ALGORITHM")
    information_list.append(f"Report {dateandtime}")
    information_list.append(f"\nNumber of runs: {number_of_runs}\n")
    information_list.append("INPUT:")
    information_list.append(f"Number of items in main database: {dimension}")
    information_list.append(f"Number of subproblems: {number_of_subproblems}")
    information_list.append(f"Number of items per sub-database: {sub_dimension}")
    information_list.append(f"Number of qubits per sub-problem: {qubits_per_subproblem}")
    if type(manual_iter) == int:
        information_list.append(f"Grover iterations per sub-circuit: {manual_iter}")
    information_list.append(f"Shots per sub-problem: {shots}\n")
    if simulation and not noise and not diff_backend:
        information_list.append(f"QUANTUM COMPUTER: ideal Qasm Simulator")
    if simulation and noise and not diff_backend:
        information_list.append(f"QUANTUM COMPUTER: {backend.configuration().backend_name}")
    if not simulation and not diff_backend:
        information_list.append(f"QUANTUM COMPUTER: {backend}")

    opening_information = "\n".join(information_list)
    return(opening_information)


def write_report(information_to_write, to_file, filename):

    print(information_to_write)

    if to_file:
        with open(filename, mode='a') as report_file:
            print(information_to_write, file=report_file)