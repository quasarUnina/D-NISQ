from qiskit import *
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import *
import matplotlib.pyplot as plt
from qiskit import IBMQ


def check_job_status(job, subproblem_label, thread_pool):

    running = False
    done = False

    if thread_pool:
        use_the_word = "Sub-problem"
    else:
        use_the_word = "Thread"

    while True:
        job_status = job.status()
        if job_status == JobStatus.RUNNING and not running:
            running = True
            print(f"\n{use_the_word} {subproblem_label + 1}: {job_status}")
        if job_status == JobStatus.DONE and not done:
            done = True
            print(f"\n{use_the_word} {subproblem_label + 1}: {job_status}")
            break


def quantum_circuit_executer(subproblem_label, circuit, simulation, noise, shots, backend, mock_backend_list,
                             destination_list, thread_pool, provider):
    if thread_pool:
        if simulation:
            if noise:
                print(f"Sub-problem {subproblem_label + 1}: taking backend from mock-backend list...")
                simulator = mock_backend_list[subproblem_label]
                print(f"Sub-problem {subproblem_label + 1}: backend taken")
            else:
                simulator = Aer.get_backend('qasm_simulator')
            job = execute(circuit, shots=shots, backend=simulator)
        else:
            
            provider = provider
            qcomp = provider.get_backend(backend)
            job = execute(circuit, backend=qcomp, shots=shots)

            # Attenzione perché se si accende questo non si riesce a sapere il job status in tempo reale
            # job_monitor(job)

        check_job_status(job, subproblem_label, thread_pool)

        result = job.result()
        return result

    else:
        if simulation:
            if noise:
                print(f"Thread {subproblem_label + 1}: taking backend from mock-backend list...")
                simulator = mock_backend_list[subproblem_label]
                print(f"Thread {subproblem_label + 1}: backend taken")
            else:
                simulator = Aer.get_backend('qasm_simulator')
            job = execute(circuit, shots=shots, backend=simulator)
        else:
            
            provider = provider
            qcomp = provider.get_backend(backend)
            job = execute(circuit, backend=qcomp, shots=shots)

            # Attenzione perché se si accende questo non si riesce a sapere il job status in tempo reale
            # job_monitor(job)

        check_job_status(job, subproblem_label, thread_pool)

        destination_list[subproblem_label] = job.result()


def circuit_draw(circuit):

    circuit_drawer(circuit, output='mpl',)
    plt.show()


def show_data(number_of_subproblems, sub_counts):

    legend = []
    for i in range(number_of_subproblems):
        subproblem_number = f"Sub-problem  {i + 1}"
        legend.append(subproblem_number)

    plot_histogram(sub_counts, legend=legend, bar_labels=False)
    plt.show()

    for l in range(number_of_subproblems):
        plot_histogram(sub_counts[l])
        plt.show()








