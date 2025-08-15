import numpy as np
from collections import defaultdict, deque
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback

import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# -----------------------
# Problem definition
# -----------------------
class FixedPlatformProblem(ElementwiseProblem):
    def __init__(self):
        self.n_tasks = 4
        self.n_pe = 4

        # PE types and parameters
        self.pe_power = np.array([2.2, 1.2, 1.8, 1.2])  # Watts
        self.exec_time = np.array([
            [0.9, 1.4, 0.7, 1.4],
            [1.1, 1.0, 0.6, 1.0],
            [0.8, 1.2, 0.9, 1.2],
            [1.3, 0.9, 0.7, 0.9]
        ])  # seconds

        self.edges = [
            (0, 1, 8),
            (0, 2, 6),
            (1, 2, 4),
            (1, 3, 10),
            (2, 3, 6),
        ]

        self.bus_bw = 50.0  # kB/s
        self.bus_power = 0.5

        super().__init__(n_var=self.n_tasks, n_obj=3, n_ieq_constr=0, xl=0, xu=self.n_pe - 1)

    def comm_delay(self, mapping, u, v):
        src, dst = mapping[u], mapping[v]
        if src == dst:
            return 0.0
        size_kB = next((s for uu, vv, s in self.edges if uu == u and vv == v), 0.0) # find size of edge u->v
        return size_kB / self.bus_bw # time in seconds

    def topo_order(self): # 
        # Kahn's algorithm for topological sorting
        indeg = [0]*self.n_tasks  # num of input edges
        g = defaultdict(list)  # adj list , list of successors
        for u, v, _ in self.edges:
            g[u].append(v) # add edge u->v
            indeg[v] += 1 # increment indegree of v
        q = deque([i for i in range(self.n_tasks) if indeg[i] == 0]) # start with tasks that have no incoming edges or start from tasks that dont have any precedors
        order = [] 
        while q: # process tasks in topological order
            u = q.popleft()
            order.append(u)
            # for each successor of u, decrement indegree and if it becomes 0, add to queue 
            for w in g[u]: 
                indeg[w] -= 1 
                if indeg[w] == 0:
                    q.append(w)
        return order

    def compute_t_b_levels(self, mapping):
        exec_times_map = np.array([self.exec_time[t, mapping[t]] for t in range(self.n_tasks)]) # execution times for each task on its assigned PE based on mapping matrix

        # Create predecessors and successors lists
        # preds[i] = [(u, size_kB), ...] where u -> i
        # succs[i] = [(v, size_kB), ...] where i -> v
        preds = {i: [(u, s) for u, v, s in self.edges if v == i] for i in range(self.n_tasks)}
        succs = {i: [(v, s) for u, v, s in self.edges if u == i] for i in range(self.n_tasks)}
        order = self.topo_order()

        tlevel = np.zeros(self.n_tasks) # tlevel means the earliest time a task can start execution
        for u in order:
            if preds[u]:
                tlevel[u] = max(tlevel[p] + exec_times_map[p] + self.comm_delay(mapping, p, u) for p, _ in preds[u])
            else:
                tlevel[u] = 0.0

        blevel = np.zeros(self.n_tasks) # blevel means the latest time a task can finish execution without delaying the overall schedule
        for u in reversed(order):
            if succs[u]:
                blevel[u] = exec_times_map[u] + max(blevel[s] + self.comm_delay(mapping, u, s) for s, _ in succs[u])
            else:
                blevel[u] = exec_times_map[u]

        return tlevel, blevel

    def _evaluate(self, x, out, *args, **kwargs):
        mapping = np.asarray(np.rint(x), dtype=int) # ensure mapping is integer type

        # Priorities from t-level / b-level
        tlevel, blevel = self.compute_t_b_levels(mapping)
        priority_list = sorted(range(self.n_tasks), key=lambda t: (-tlevel[t], -blevel[t]))

        pe_ready = np.zeros(self.n_pe)
        finish = np.zeros(self.n_tasks)
        total_energy = 0.0
        total_comm_energy = 0.0
        preds = {i: [(u, s) for u, v, s in self.edges if v == i] for i in range(self.n_tasks)}

        for t in priority_list:
            pe = mapping[t]
            est = pe_ready[pe]
            for p, size_kB in preds[t]:
                pred_ready = finish[p]
                if mapping[p] != pe:
                    pred_ready += size_kB / self.bus_bw
                    total_comm_energy += self.bus_power * (size_kB / self.bus_bw)
                est = max(est, pred_ready)
            rt = self.exec_time[t, pe]
            st = est
            ft = st + rt
            finish[t] = ft
            pe_ready[pe] = ft
            total_energy += rt * self.pe_power[pe]

        makespan = float(np.max(finish))
        energy = float(total_energy + total_comm_energy)
        cost = energy * 10.0  

        out["F"] = np.array([cost, energy, makespan], dtype=float)


# -----------------------
# Operators
# -----------------------
class MappingSampling(Sampling): # sampling operator for generating initial mappings
    def _do(self, problem, n_samples, **kwargs):
        return np.random.randint(0, problem.n_pe, size=(n_samples, problem.n_tasks))

class MappingCrossover(Crossover): # Uniform Crossover
    def __init__(self, p=0.5):
        super().__init__(2, 2)
        self.p = p
    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape # n_matings is number of pairs being mated
        Y = np.empty_like(X)
        for k in range(n_matings):
            a = np.rint(X[0, k]).astype(int) # parent 1 - X[0, k] is the first parent in the k-th mating pair - k is the index of the mating pair
            b = np.rint(X[1, k]).astype(int) # parent 2 - X[1, k] is the second parent in the k-th mating pair
            # print("Crossover parents:", a, b)
            mask = np.random.rand(n_var) < self.p
            Y[0, k] = np.where(mask, a, b)
            Y[1, k] = np.where(mask, b, a)
        return Y

class MappingMutation(Mutation):
    def __init__(self, p_gene=0.2): # 0.2 means 20% chance of mutation per gene
        super().__init__()
        self.p_gene = p_gene
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            xi = np.rint(X[i]).astype(int)
            mask = np.random.rand(problem.n_tasks) < self.p_gene # decide which genes to mutate
            xi[mask] = np.random.randint(0, problem.n_pe, size=np.sum(mask)) # replace with random PE
            X[i] = xi # update the individual
        return X

class MappingDuplicate(ElementwiseDuplicateElimination): # to eliminate duplicate solutions
    def is_equal(self, a, b):
        return np.array_equal(np.rint(a.X).astype(int), np.rint(b.X).astype(int))


# -----------------------
# Callback for tracking fronts
# -----------------------
class ParetoTrace(Callback): # callback is used to track the Pareto fronts during optimization
    def __init__(self):
        super().__init__()
        self.fronts = []
        self.gens = []
    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        nd_idx = []
        for i in range(len(F)):
            if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F)) if j != i):
                nd_idx.append(i)
        self.fronts.append(F[nd_idx])
        self.gens.append(algorithm.n_gen)


# -----------------------
# Run optimization
# -----------------------
problem = FixedPlatformProblem()
algorithm = NSGA2(
    pop_size=20,
    sampling=MappingSampling(),
    crossover=MappingCrossover(),
    mutation=MappingMutation(),
    eliminate_duplicates=MappingDuplicate()
)
trace = ParetoTrace()

res = minimize(problem, algorithm, ('n_gen', 15), seed=1, callback=trace, verbose=True)

# -----------------------
# Final Pareto front plot
# -----------------------

# Print chromosomes (decision vectors) of Pareto front
X = res.X
F = res.F  # ensure F is in scope
nds = NonDominatedSorting()
nd_idx = nds.do(F, only_non_dominated_front=True)
# Ensure integer mapping even if operators passed floats
if isinstance(X, np.ndarray) and X.dtype != object:
    X_int = np.rint(X).astype(int)
else:
    # handle possible object dtype (list of arrays)
    X_int = np.array([np.rint(np.asarray(xx)).astype(int) for xx in X])

print("\n=== Final Pareto Front Chromosomes (task -> PE) ===")
for k, i in enumerate(nd_idx):
    c, e, l = F[i]
    print(f"Sol#{k}: map={X_int[i].tolist()}   "
          f"F=[Cost={c:.4f}, Energy={e:.4f}, Latency={l:.4f}]")



# F = res.F  # (n_points, 3) with [Cost, Energy, Latency]
F = res.F

# find non-dominated indices
nds = NonDominatedSorting()
nd_idx = nds.do(F, only_non_dominated_front=True)
dom_idx = np.setdiff1d(np.arange(len(F)), nd_idx)

# plot
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# (optional) show dominated points in light gray
if len(dom_idx) > 0:
    ax.scatter(F[dom_idx, 0], F[dom_idx, 1], F[dom_idx, 2],
               s=35, alpha=0.35, label="Dominated")

# non-dominated (Pareto front) in red
ax.scatter(F[nd_idx, 0], F[nd_idx, 1], F[nd_idx, 2],
           s=60, label="Non-dominated")

# annotate ND points as Sol#k
for k, i in enumerate(nd_idx):
    ax.text(F[i, 0], F[i, 1], F[i, 2], f"Sol#{k}", fontsize=9)

ax.set_title("Final Pareto Front (annotated)")
ax.set_xlabel("Cost")
ax.set_ylabel("Energy")
ax.set_zlabel("Latency")
ax.legend()
plt.tight_layout()
plt.show()
