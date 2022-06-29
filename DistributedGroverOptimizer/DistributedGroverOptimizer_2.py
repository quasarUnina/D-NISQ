from DistributedGroverOptimizer_1 import *


# DISTRIBUTED GROVER OPTIMIZER ----------------------------------------------------------------------------------------
def Distributed_Grover_Optimizer(RUN, NUMBER_OF_SUBPROBLEMS, N_VAR, OBJECTIVE_FUNCTION, MAX_ITERATIONS_LIST,
                                 REPORT_FILE=False, FILE_NAME=None, SHOW_GAS_STEPS=False, NOISE=False,
                                 MOCK_BACKEND=FakeMelbourne()):
    assert NUMBER_OF_SUBPROBLEMS <= pow(2,
                                        N_VAR - 2), f"The objective function has {N_VAR} variables, so the number of " \
                                                    f"subproblems must be smaller or equal to {int(pow(2, N_VAR - 2))}."
    backend = Aer.get_backend('qasm_simulator')
    if not NOISE:
        qi = backend
    else:
        device = QasmSimulator.from_backend(MOCK_BACKEND)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates
        qi = QuantumInstance(backend=backend, basis_gates=basis_gates, coupling_map=coupling_map,
                             noise_model=noise_model,
                             shots=1000)

    max_iterations = len(MAX_ITERATIONS_LIST)
    constant_bits = int(math.log(NUMBER_OF_SUBPROBLEMS, 2))
    threadlock = Lock()
    results = [None for _ in range(NUMBER_OF_SUBPROBLEMS)]

    if NUMBER_OF_SUBPROBLEMS == 1:
        main_qp, exact_result, int_exact_result, min_value = qubo_function(OBJECTIVE_FUNCTION, N_VAR,
                                                                           NUMBER_OF_SUBPROBLEMS,
                                                                           constant_bits)
    else:
        main_qp, exact_result, int_exact_result, min_value, arrays_for_fixed_values, SUBPROGRAMS = \
            qubo_function(OBJECTIVE_FUNCTION, N_VAR, NUMBER_OF_SUBPROBLEMS, constant_bits)

    average_times = []
    max_times = []
    probabilities = []
    average_grover_executions = []
    max_grover_executions = []
    average_m_parameters = []
    max_m_parameters = []
    average_on_runs_i = []

    # Loop over numbers of max iterations
    for n_iter in MAX_ITERATIONS_LIST:
        successes = 0
        times = []
        worst_thread_grover_executions = []
        worst_thread_m = []

        for run in range(RUN):
            GROVER_ALGORITHM_EXECUTIONS = [0 for _ in range(NUMBER_OF_SUBPROBLEMS)]
            M = [1 for _ in range(NUMBER_OF_SUBPROBLEMS)]
            thresholds = []
            
            print(f"\n\033[1;33m DISTRIBUTED GROVER OPTIMIZER\033[0m"
                   f"\n\033[33m Max iterations without improvement: {n_iter}\n Run: {run + 1}\033[0m")

            if NUMBER_OF_SUBPROBLEMS == 1:
                thread_list = [Thread(target=grover_optimizer_function,
                                      args=(
                                      NUMBER_OF_SUBPROBLEMS, l, main_qp, results, n_iter, backend, qi, SHOW_GAS_STEPS,
                                      threadlock, thresholds)) for l in range(NUMBER_OF_SUBPROBLEMS)]
            else:
                thread_list = [Thread(target=grover_optimizer_function,
                                      args=(NUMBER_OF_SUBPROBLEMS, l, SUBPROGRAMS[l], results, n_iter, backend, qi,
                                            SHOW_GAS_STEPS, threadlock, thresholds)) for l in
                               range(NUMBER_OF_SUBPROBLEMS)]

            start = perf_counter()
            print("\nStarting threads...")
            for thread in thread_list:
                thread.start()

            for thread in thread_list:
                thread.join()
            print("\n\n\nAll threads are done!")
            stop = perf_counter()
            time_taken = stop - start
            times.append(time_taken)

            x_values = [result.x for result in results]
            f_values = [result.fval for result in results]
            grover_algorithm_executions = [result.grover_executions for result in results]
            m_parameters = [result.max_m for result in results]

            if NUMBER_OF_SUBPROBLEMS == 1:
                assembled_x = results[0].x
            else:
                # Find the label of the subproblem with smallest fval
                i = f_values.index(min(f_values))
                first_piece = arrays_for_fixed_values[i]
                second_piece = [x_values[i][k] for k in range(constant_bits, N_VAR)]
                if SHOW_GAS_STEPS is True:
                    print(first_piece)
                    print(second_piece)
                assembled_x = np.append(first_piece, second_piece)

            print("\nDISTRIBUTED GROVER OPTIMIZER RESULT FOR MAIN Q.U.B.O. PROBLEM:")
            print("x = {}".format(assembled_x))
            print("fval = {}".format(min(f_values)))

            # Compare the f minimum value with the exact result
            fval_comparison = min(f_values) == exact_result.fval
            success = fval_comparison

            if success:
                successes += 1
                if SHOW_GAS_STEPS is True:
                
                    print("\nSuccess!")
            else:
                if SHOW_GAS_STEPS is True:
                
                    print("\nFail!")
            if SHOW_GAS_STEPS is True:
            
                print(f"\nThresholds found in the process:"
                      f"\n{thresholds}")
                print("\nMaximum \"m\" parameters reached:")
                for t in range(NUMBER_OF_SUBPROBLEMS):
                    print(f"Thread {t + 1}: {m_parameters[t]}")
                print("\nExecutions of Grover's algorithm:")
                for t in range(NUMBER_OF_SUBPROBLEMS):
                    print(f"Thread {t + 1}: {grover_algorithm_executions[t]}")

            # Worst thread scores for this single run
            worst_thread_grover_executions.append(max(grover_algorithm_executions))
            worst_thread_m.append(max(m_parameters))
            average_i = mean(grover_algorithm_executions)

        average_times.append(round(mean(times), 3))
        max_times.append(round(max(times), 3))

        probability = successes / RUN
        probabilities.append(probability)

        average_worstsinglethread_grover_executions = round(mean(worst_thread_grover_executions))
        average_grover_executions.append(average_worstsinglethread_grover_executions)
        max_worstsinglethread_grover_executions = max(worst_thread_grover_executions)
        max_grover_executions.append(max_worstsinglethread_grover_executions)

        average_on_runs_i.append(average_i)

        average_worstsinglethread_m = round(mean(worst_thread_m))
        average_m_parameters.append(average_worstsinglethread_m)
        max_worstsinglethread_m = max(worst_thread_m)
        max_m_parameters.append(max_worstsinglethread_m)

        # Report
        if REPORT_FILE:
            titles = ["Max iterations", "Success rate", "Av.i"]
            report_to_file(FILE_NAME, MAX_ITERATIONS_LIST, MAX_ITERATIONS_LIST.index(n_iter) + 1, 
                           probabilities,  average_on_runs_i)
            report_to_excel(FILE_NAME, MAX_ITERATIONS_LIST, MAX_ITERATIONS_LIST.index(n_iter) + 1, 
                           probabilities,  average_on_runs_i)

    print_report(max_iterations, MAX_ITERATIONS_LIST,  probabilities,
                  average_on_runs_i)
        
    return probabilities, average_on_runs_i


