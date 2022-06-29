from threading import *
from concurrent.futures import *
from datetime import datetime
from statistics import mean

from GroverCircuitBuilder import *
from QuantumComputing import *
from DistributedGroverAlgorithm_1 import *

# GENERAL SETTINGS ----------------------------------------------------------------------------------------------------

dateandtime = datetime.now().strftime("%d.%m.%Y  %H.%M.%S")

RUN = 1                                  # Iterations of the whole experiment
REPORT_FILE = True                       # True to create a .txt report file
FILE_PATH = f"C:\\Users\\green\\Desktop\\"
FILE_NAME = f"Report_{dateandtime}.txt"

# DATABASE SETTINGS ---------------------------------------------------------------------------------------------------

N = 8                                    # The database will be a list of N string elements (must be a power of 2)
WINNER = "1111111111"                    # Winner string (arbitrary)
OTHER = "0000000000"                     # Non-winner strings to fill the list (arbitrary)
WINNER_INDEX = 5                         # Winner location in the list (arbitrary)

# PARALLELISM SETTINGS ------------------------------------------------------------------------------------------------

NUMBER_OF_SUBPROBLEMS = 2                # Set 1 to perform a standard search without parallelism
THREAD_POOL = False                      # True to implement parallelism with the concurrent.futures module (otherwise
                                         # the threading module will be used)
MAX_WORKERS = 32                         # Maximum number of threads created by the ThreadPoolExecutor class

# GROVER'S ALGORITHM SETTINGS -----------------------------------------------------------------------------------------

MANUAL_ITERATIONS = False                # Either False or an integer to set the number of Grover iterations manually
HISTOGRAMS = False                       # True to plot counts histograms at the end of each Grover search

# BACKEND SETTINGS ----------------------------------------------------------------------------------------------------

SIMULATION = False                       # True for simulators, False for real quantum computers
NOISE = True                             # True for simulation with a noise model, False for ideal simulation
BACKEND = 'ibmq_manila'                  # EXAMPLE: FakeValencia() for mock backend, 'ibmq_valencia' for real device
DIFFERENT_BACKENDS = False               # True to assign a different backend to each subproblem
SHOTS = 1024                             # Number of shots for each backend

# DISTRIBUTED GROVER'S SEARCH -----------------------------------------------------------------------------------------

max_times = []
successes = 0
sub_N = int(N / NUMBER_OF_SUBPROBLEMS)

assert N % NUMBER_OF_SUBPROBLEMS == 0, "Please choose the number of subproblems as a divisor of N."

sub_n = int(log(sub_N, 2))
THREADLOCK = Lock()

write_report(report_opening(dateandtime, RUN, N, NUMBER_OF_SUBPROBLEMS, sub_N, sub_n, MANUAL_ITERATIONS, SHOTS,
                           SIMULATION, NOISE, DIFFERENT_BACKENDS, BACKEND), REPORT_FILE, FILE_PATH + FILE_NAME)

if SIMULATION and NOISE:
    print("\nCreating a mock-backend list...")
    MOCK_BACKEND_LIST = [QasmSimulator.from_backend(BACKEND) for _ in range(NUMBER_OF_SUBPROBLEMS)]
    print("\nMock-backend list created.")
else:
    MOCK_BACKEND_LIST = None

if DIFFERENT_BACKENDS:
    backend_list, backend_information = ask_for_backends(NUMBER_OF_SUBPROBLEMS)
    write_report(backend_information, REPORT_FILE, FILE_PATH + FILE_NAME)

database = create_database(N, WINNER, OTHER, WINNER_INDEX)
sub_databases, winner_coordinates = create_sub_databases(database, NUMBER_OF_SUBPROBLEMS, sub_N, WINNER)
sub_basis = create_sub_basis(NUMBER_OF_SUBPROBLEMS, sub_N, sub_n)
sub_oracles = create_sub_oracles(NUMBER_OF_SUBPROBLEMS, sub_N, winner_coordinates)
sub_diffusers = create_sub_diffusers(NUMBER_OF_SUBPROBLEMS, sub_N, sub_basis)

sub_grover_circuits = [grover_circuit_builder(sub_N, sub_n, sub_oracles[i], sub_diffusers[i], MANUAL_ITERATIONS)
                       for i in range(NUMBER_OF_SUBPROBLEMS)]



for run in range(RUN):

    run_information = f"\n\n\n------------------------ RUN {run+1} -----------------------------\n"
    write_report(run_information, REPORT_FILE, FILE_PATH + FILE_NAME)

    results = distributed_grover_search(NUMBER_OF_SUBPROBLEMS, sub_grover_circuits, SIMULATION, NOISE, SHOTS, BACKEND,
                                        MOCK_BACKEND_LIST, THREAD_POOL, MAX_WORKERS)

    sub_counts, counts_information = get_counts(results, NUMBER_OF_SUBPROBLEMS, THREAD_POOL)
    write_report(counts_information, REPORT_FILE, FILE_PATH + FILE_NAME)

    execution_times, max_time, times_information = get_times(results, NUMBER_OF_SUBPROBLEMS, THREAD_POOL)
    max_times.append(max_time)
    write_report(times_information, REPORT_FILE, FILE_PATH + FILE_NAME)

    success, success_information = verify_success(NUMBER_OF_SUBPROBLEMS, sub_N, sub_n, sub_counts, WINNER_INDEX)
    write_report(success_information, REPORT_FILE, FILE_PATH + FILE_NAME)

    if success:
        successes += 1

    write_report(current_situation(run + 1, max_times, successes), REPORT_FILE, FILE_PATH + FILE_NAME)

    if HISTOGRAMS:
        show_data(NUMBER_OF_SUBPROBLEMS, sub_counts)

write_report("\n\n\n\n", REPORT_FILE, FILE_PATH + FILE_NAME)
