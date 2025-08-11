import numpy as np  # Import NumPy for numerical operations and array handling
from pymoo.algorithms.moo.nsga2 import NSGA2  # Import NSGA-II algorithm for multi-objective optimization
from pymoo.core.problem import ElementwiseProblem  # Import base class for defining optimization problems
from pymoo.core.sampling import Sampling  # Import base class for initial population sampling
from pymoo.core.crossover import Crossover  # Import base class for crossover operators
from pymoo.core.mutation import Mutation  # Import base class for mutation operators
from pymoo.core.duplicate import ElementwiseDuplicateElimination  # Import base class for duplicate elimination
from pymoo.optimize import minimize  # Import function to run the optimization
from pymoo.visualization.scatter import Scatter  # Import visualization tool for plotting Pareto front

# Define the problem class for embedded system optimization
class EmbeddedSystemProblem(ElementwiseProblem):
    def __init__(self, n_tasks=10, max_pes=5, n_pe_types=3, n_buses=2):
        # Initialize the problem with 1 variable, 3 objectives, and 2*max_pes constraints
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=2 * max_pes)   # defines two inequality constraints (memory and buffer violations) per PE to track local violations independently
        self.n_tasks = n_tasks  # Number of tasks to schedule
        self.max_pes = max_pes  # Maximum number of processing elements (PEs)
        self.n_pe_types = n_pe_types  # Number of possible PE types
        self.n_buses = n_buses  # Number of buses for PE connections
        # Generate random cost matrix for each task and PE type
        self.task_costs = np.random.randint(50, 150, size=(n_tasks, n_pe_types))
        # Generate random energy consumption matrix for each task and PE type
        self.task_energy = np.random.randint(100, 500, size=(n_tasks, n_pe_types))
        # Generate random memory usage matrix for each task and PE type
        self.task_memory = np.random.randint(50, 200, size=(n_tasks, n_pe_types))
        # Generate random execution time matrix for each task and PE type
        self.task_execution_time = np.random.randint(10, 100, size=(n_tasks, n_pe_types))
        # Define memory limits for each PE (constant for all PEs)
        self.memory_limits = np.array([500] * max_pes) # array with len = max_pes
        # Define buffer limits for each PE (constant for all PEs)
        self.buffer_limits = np.array([1000] * max_pes)
        # Generate random task dependency graph (0 or 1 for dependencies)
        self.task_graph = np.random.randint(0, 2, size=(n_tasks, n_tasks))  # matrix
        # Ensure task graph is acyclic by keeping only lower triangular part
        self.task_graph = np.tril(self.task_graph, -1) # for keeping dependencies
        # Generate random communication data volume for dependent tasks
        self.comm_data = np.random.randint(10, 50, size=(n_tasks, n_tasks)) * self.task_graph
        # Generate random deadlines for each task
        self.task_deadlines = np.random.randint(50, 200, size=n_tasks)

    def _evaluate(self, x, out, *args, **kwargs):
        # Extract solution components: task-to-PE mapping, PE types, and PE-to-bus assignments
        mapping, pe_types, pe_bus = x[0] # unpacking
        n_pes = len(pe_types)  # Number of PEs in the current solution
        # Ensure mapping values are valid (within [0, n_pes-1])
        mapping = np.clip(mapping, 0, n_pes - 1)

        # 1. Calculate total cost by summing costs of tasks on their assigned PEs
        total_cost = sum(self.task_costs[i, pe_types[mapping[i]]] for i in range(self.n_tasks))
        
        ##########  TEST  ##########

        # 2. Calculate total energy by summing energy consumption of tasks on their assigned PEs
        total_energy = sum(self.task_energy[i, pe_types[mapping[i]]] for i in range(self.n_tasks))
        
        ##########  TEST  ##########

        # 3. Static List Scheduling to calculate latency (makespan)
        # Compute indegree (number of dependencies) for each task
        indegree = np.sum(self.task_graph, axis=0)  # axis=0 means sum of culomns
        # Sort tasks by indegree to prioritize tasks with fewer dependencies
        priority_list = np.argsort(indegree)
        # Initialize array to track when each PE is available
        pe_available_time = np.zeros(n_pes) # after assigning tasks -> values update
        # Initialize array to store finish times of tasks
        finish_times = np.zeros(self.n_tasks)
        # Initialize array to track scheduled tasks
        scheduled = np.zeros(self.n_tasks, dtype=bool)
        # Iterate through tasks in priority order
        for task_idx in priority_list:
            if scheduled[task_idx]:  # Skip already scheduled tasks
                continue
            start_time = 0  # Initialize start time for the task
            # Find the maximum finish time of dependent tasks
            for j in range(self.n_tasks):
                if self.task_graph[j, task_idx]:  # If task j is a dependency
                    start_time = max(start_time, finish_times[j])
            pe = mapping[task_idx]  # Get the PE assigned to the task
            # Get execution time of the task on the assigned PE's type
            exec_time = self.task_execution_time[task_idx, pe_types[pe]]
            # Task starts when both dependencies and PE are available
            start_time = max(start_time, pe_available_time[pe])
            # Calculate finish time of the task
            finish_times[task_idx] = start_time + exec_time
            # Update PE availability time
            pe_available_time[pe] = finish_times[task_idx]
            scheduled[task_idx] = True  # Mark task as scheduled
        # Total latency is the maximum finish time across all tasks
        total_latency = max(finish_times)

        # 4. Calculate memory violation for each PE
        memory_usage = np.zeros(self.max_pes)  # Initialize memory usage for max_pes
        # Sum memory usage for tasks assigned to each PE
        for i in range(self.n_tasks):
            if mapping[i] < n_pes:  # Only consider valid PEs
                memory_usage[mapping[i]] += self.task_memory[i, pe_types[mapping[i]]]
        # Calculate memory violation (excess over limit) for each PE
        memory_violation = np.array([max(0, memory_usage[i] - self.memory_limits[i]) for i in range(self.max_pes)])

        # 5. Calculate buffer violation for each PE
        buffer_usage = np.zeros(self.max_pes)  # Initialize buffer usage for max_pes
        # Sum buffer usage for communication between tasks on different PEs
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if self.task_graph[i, j] and mapping[i] != mapping[j] and mapping[i] < n_pes and mapping[j] < n_pes:
                    buffer_usage[mapping[i]] += self.comm_data[i, j]  # Outgoing data
                    buffer_usage[mapping[j]] += self.comm_data[i, j]  # Incoming data
        # Calculate buffer violation (excess over limit) for each PE
        buffer_violation = np.array([max(0, buffer_usage[i] - self.buffer_limits[i]) for i in range(self.max_pes)])

        # Store objectives (cost, energy, latency) in output dictionary
        out["F"] = np.array([total_cost, total_energy, total_latency], dtype=float)
        # Store constraints (memory and buffer violations) in output dictionary
        out["G"] = np.concatenate([memory_violation, buffer_violation])

# Custom sampling class to generate initial population
class TaskSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # Initialize array to store solutions
        X = np.full((n_samples, 1), None, dtype=object)
        # Generate n_samples solutions
        for i in range(n_samples):
            # Randomly choose number of PEs for this solution
            n_pes = np.random.randint(1, problem.max_pes + 1)
            # Generate random task-to-PE mapping
            mapping = np.random.randint(0, n_pes, size=problem.n_tasks)
            # Generate random PE types
            pe_types = np.random.randint(0, problem.n_pe_types, size=n_pes)
            # Generate random PE-to-bus assignments
            pe_bus = np.random.randint(0, problem.n_buses, size=n_pes)
            # Store solution as a tuple (mapping, pe_types, pe_bus)
            X[i, 0] = (mapping, pe_types, pe_bus)
        return X  # X means solutions

# Custom crossover class for task mappings and PE types
class TaskCrossover(Crossover):  # # Crossover operator: swaps PE type and one task mapping between parents

    def __init__(self):
        # Initialize crossover with 2 parents and 2 offspring
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):  # X means parents
        # Get dimensions of input population
        _, n_matings, _ = X.shape  # in pymoo -> X.shape == (n_parents, n_matings, n_genes)  or  (2, n_matings, ...)
        # Initialize array to store offspring
        Y = np.full_like(X, None, dtype=object)
        # Perform crossover for each pair of parents
        for k in range(n_matings):
            # Extract components of parent solutions
            mapping_a, pe_types_a, pe_bus_a = X[0, k, 0]
            mapping_b, pe_types_b, pe_bus_b = X[1, k, 0]
            n_pes_a, n_pes_b = len(pe_types_a), len(pe_types_b)
            # Use minimum number of PEs to ensure compatibility
            n_pes = min(n_pes_a, n_pes_b)
            # Ensure mappings are valid for the number of PEs
            mapping_a = np.clip(mapping_a, 0, n_pes - 1).copy()
            mapping_b = np.clip(mapping_b, 0, n_pes - 1).copy()
            # Truncate PE types and bus assignments to n_pes
            pe_types_a = pe_types_a[:n_pes].copy()
            pe_types_b = pe_types_b[:n_pes].copy()
            pe_bus_a = pe_bus_a[:n_pes].copy()
            pe_bus_b = pe_bus_b[:n_pes].copy()
            # Select a random PE for crossover
            pe_idx = np.random.randint(0, n_pes)
            # Swap PE types between parents
            pe_types_a[pe_idx], pe_types_b[pe_idx] = pe_types_b[pe_idx], pe_types_a[pe_idx]
            # Find tasks assigned to the selected PE in each parent
            tasks_on_pe_a = np.where(mapping_a == pe_idx)[0]
            tasks_on_pe_b = np.where(mapping_b == pe_idx)[0]
            # If both PEs have tasks, swap one task between parents
            if len(tasks_on_pe_a) > 0 and len(tasks_on_pe_b) > 0:
                task_a = np.random.choice(tasks_on_pe_a)
                task_b = np.random.choice(tasks_on_pe_b)
                mapping_a[task_a], mapping_b[task_b] = mapping_b[task_b], mapping_a[task_a]
            # Store offspring solutions
            Y[0, k, 0] = (mapping_a, pe_types_a, pe_bus_a)
            Y[1, k, 0] = (mapping_b, pe_types_b, pe_bus_b)
        return Y

# Custom mutation class implementing article-specific operators
class TaskMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        # Iterate through each solution in the population
        for i in range(len(X)):
            # Extract solution components
            mapping, pe_types, pe_bus = X[i, 0]
            n_pes = len(pe_types)  # Number of PEs in the solution
            # Randomly select a mutation operator with specified probabilities
            mutation_type = np.random.choice(
                ['change_pe', 'add_pe', 'remove_pe', 'random_reassign', 'heuristic_reassign'],
                p=[0.2, 0.2, 0.2, 0.3, 0.1]
            )

            if mutation_type == 'change_pe':
                # Change PE: Randomly change the type and bus of a PE
                if n_pes > 0:
                    pe_idx = np.random.randint(0, n_pes)  # Select a random PE
                    # Assign a new random type to the PE
                    pe_types[pe_idx] = np.random.randint(0, problem.n_pe_types)
                    # Assign a new random bus to the PE
                    pe_bus[pe_idx] = np.random.randint(0, problem.n_buses)

            elif mutation_type == 'add_pe':
                # Add PE: Add a new PE and reassign tasks to it
                if n_pes < problem.max_pes:
                    # Create a new PE with random type
                    new_pe_type = np.random.randint(0, problem.n_pe_types)
                    # Assign the new PE to a random bus
                    new_pe_bus = np.random.randint(0, problem.n_buses)
                    # Append new PE type and bus to arrays
                    pe_types = np.append(pe_types, new_pe_type)
                    pe_bus = np.append(pe_bus, new_pe_bus)
                    n_pes += 1  # Increment number of PEs
                    # Randomly select number of tasks to reassign
                    n_tasks_to_reassign = np.random.randint(1, problem.n_tasks + 1)
                    # Select tasks to reassign to the new PE
                    tasks_to_reassign = np.random.choice(problem.n_tasks, n_tasks_to_reassign, replace=False)
                    # Reassign selected tasks to the new PE
                    for task in tasks_to_reassign:
                        mapping[task] = n_pes - 1

            elif mutation_type == 'remove_pe':
                # Remove PE: Remove a PE and reassign its tasks
                if n_pes > 1:
                    pe_idx = np.random.randint(0, n_pes)  # Select a random PE to remove
                    # Find tasks assigned to the PE
                    tasks_to_reassign = np.where(mapping == pe_idx)[0]
                    # Remove the PE from types and bus arrays
                    pe_types = np.delete(pe_types, pe_idx)
                    pe_bus = np.delete(pe_bus, pe_idx)
                    n_pes -= 1  # Decrement number of PEs
                    # Reassign tasks to remaining PEs
                    for task in tasks_to_reassign:
                        mapping[task] = np.random.randint(0, n_pes)

            elif mutation_type == 'random_reassign':
                # Randomly Re-assign Task: Move 1-4 tasks to another PE
                n_tasks_to_move = np.random.randint(1, 5)  # Select number of tasks to move
                # Randomly select tasks to reassign
                tasks = np.random.choice(problem.n_tasks, n_tasks_to_move, replace=False)
                # Reassign each task to a random PE
                for task in tasks:
                    mapping[task] = np.random.randint(0, n_pes)

            elif mutation_type == 'heuristic_reassign':
                # Heuristically Re-assign Task: Move a task missing deadline to a better PE
                mapping = np.clip(mapping, 0, n_pes - 1)  # Ensure mapping is valid
                # Initialize arrays for scheduling
                finish_times = np.zeros(problem.n_tasks)
                pe_available_time = np.zeros(n_pes)
                scheduled = np.zeros(problem.n_tasks, dtype=bool)
                # Compute indegree for task prioritization
                indegree = np.sum(problem.task_graph, axis=0)
                priority_list = np.argsort(indegree)
                # Schedule tasks to find finish times
                for task_idx in priority_list:
                    if scheduled[task_idx]:
                        continue
                    start_time = 0
                    # Find maximum finish time of dependencies
                    for j in range(problem.n_tasks):
                        if problem.task_graph[j, task_idx]:
                            start_time = max(start_time, finish_times[j])
                    pe = mapping[task_idx]  # Get assigned PE
                    exec_time = problem.task_execution_time[task_idx, pe_types[pe]]  # Get execution time
                    start_time = max(start_time, pe_available_time[pe])  # Consider PE availability
                    finish_times[task_idx] = start_time + exec_time  # Calculate finish time
                    pe_available_time[pe] = finish_times[task_idx]  # Update PE availability
                    scheduled[task_idx] = True  # Mark task as scheduled
                # Identify tasks missing their deadlines
                missed_deadlines = [i for i in range(problem.n_tasks) if finish_times[i] > problem.task_deadlines[i]]
                if missed_deadlines:
                    # Select a random task that misses its deadline
                    task = np.random.choice(missed_deadlines)
                    candidate_pes = []  # List to store PEs that meet the deadline
                    # Try assigning the task to each PE
                    for pe in range(n_pes):
                        temp_mapping = mapping.copy()  # Create a copy of the mapping
                        temp_mapping = np.clip(temp_mapping, 0, n_pes - 1)  # Ensure valid temporary mapping
                        temp_mapping[task] = pe  # Assign task to current PE
                        # Initialize temporary arrays for scheduling
                        temp_finish_times = np.zeros(problem.n_tasks)
                        temp_pe_available = np.zeros(n_pes)
                        temp_scheduled = np.zeros(problem.n_tasks, dtype=bool)
                        # Schedule tasks with temporary mapping
                        for t in priority_list:
                            if temp_scheduled[t]:
                                continue
                            st = max([temp_finish_times[j] for j in range(problem.n_tasks) if problem.task_graph[j, t]], default=0)
                            p = temp_mapping[t]
                            et = problem.task_execution_time[t, pe_types[p]]
                            st = max(st, temp_pe_available[p])
                            temp_finish_times[t] = st + et
                            temp_pe_available[p] = temp_finish_times[t]
                            temp_scheduled[t] = True
                        # Check if the task meets its deadline on this PE
                        if temp_finish_times[task] <= problem.task_deadlines[task]:
                            candidate_pes.append(pe)
                    # If there are candidate PEs, reassign the task to one of them
                    if candidate_pes:
                        mapping[task] = np.random.choice(candidate_pes)

            # Ensure mapping is valid after all mutations
            mapping = np.clip(mapping, 0, n_pes - 1)
            # Update solution with modified components
            X[i, 0] = (mapping, pe_types, pe_bus)
        return X

# Custom duplicate elimination class
class TaskDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        # Extract components of two solutions
        mapping_a, pe_types_a, pe_bus_a = a.X[0]
        mapping_b, pe_types_b, pe_bus_b = b.X[0]
        # Check if solutions are identical (same mapping, PE types, and bus assignments)
        return np.array_equal(mapping_a, mapping_b) and np.array_equal(pe_types_a, pe_types_b) and np.array_equal(pe_bus_a, pe_bus_b)

# Set up the NSGA-II algorithm
algorithm = NSGA2(
    pop_size=20,  # Population size of 20 individuals
    sampling=TaskSampling(),  # Use custom sampling for initial population
    crossover=TaskCrossover(),  # Use custom crossover operator
    mutation=TaskMutation(),  # Use custom mutation operator
    eliminate_duplicates=TaskDuplicateElimination()  # Use custom duplicate elimination
)

# Run the optimization
res = minimize(
    EmbeddedSystemProblem(n_tasks=10, max_pes=5, n_pe_types=3, n_buses=2),  # Define problem instance
    algorithm,  # Use NSGA-II algorithm
    ('n_gen', 50),  # Run for 50 generations
    seed=1,  # Set random seed for reproducibility
    verbose=True  # Print progress during optimization
)

# Visualize the Pareto front
plot = Scatter(
    title="Pareto Front for Embedded System Optimization",  # Set plot title
    labels=["Cost", "Energy", "Latency"]  # Label axes for objectives
)
plot.add(res.F, color="red")  # Add Pareto front points in red
plot.show()  # Display the plot

# Print the best solutions
print("Best task mappings with objectives:")
for i in range(len(res.X)):
    # Extract components of each solution
    mapping, pe_types, pe_bus = res.X[i][0]
    # Print solution details: mapping, PE types, bus assignments, and objectives
    print(f"Solution {i+1}: Mapping = {mapping}, PE Types = {pe_types}, PE Bus = {pe_bus}, Objectives (Cost, Energy, Latency) = {res.F[i]}")