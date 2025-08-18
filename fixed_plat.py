import numpy as np
from collections import defaultdict, deque
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter   # (unused; kept to match your imports)
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import matplotlib.pyplot as plt

# Try to import Hypervolume indicator (optional)
try:
    from pymoo.indicators.hv import HV   # pymoo >= 0.6
except Exception:
    HV = None


# -----------------------
# Problem definition
# -----------------------
class FixedPlatformProblem(ElementwiseProblem):
    """
    4-task DAG mapped onto a fixed 4-PE platform.
    Objectives (to minimize): [Cost, Energy, Makespan]
    Cost is set to 10 * Energy just to keep 3 objectives.
    Scheduling: static list scheduling using (t-level, b-level) priority.
    Communication: simple delay over a shared bus (no contention modeled).
    """

    def __init__(self):
        self.n_tasks = 4
        self.n_pe = 4

        # PE types and parameters (W for power; seconds for exec times)
        # Index meaning (example): [ASIC, GPP1, FPGA, GPP2]
        self.pe_power = np.array([2.2, 1.2, 1.8, 1.2])  # Watts
        self.exec_time = np.array([
            [0.9, 1.4, 0.7, 1.4],   # task 0 on PEs 0..3
            [1.1, 1.0, 0.6, 1.0],   # task 1 on PEs 0..3
            [0.8, 1.2, 0.9, 1.2],   # task 2 on PEs 0..3
            [1.3, 0.9, 0.7, 0.9]    # task 3 on PEs 0..3
        ])  # seconds

        # Directed edges with payload sizes in kB: (u, v, size_kB)
        self.edges = [
            (0, 1, 8),
            (0, 2, 6),
            (1, 2, 4),
            (1, 3, 10),
            (2, 3, 6),
        ]

        # Shared bus properties
        self.bus_bw = 50.0   # kB/s (communication bandwidth)
        self.bus_power = 0.5 # W  (consumed while transferring)

        super().__init__(n_var=self.n_tasks, n_obj=3, n_ieq_constr=0, xl=0, xu=self.n_pe - 1)

    def comm_delay(self, mapping, u, v):
        """
        Communication time (sec) for edge u->v given the mapping.
        If tasks are on the same PE, delay is zero.
        """
        src, dst = mapping[u], mapping[v]
        if src == dst:
            return 0.0
        size_kB = next((s for uu, vv, s in self.edges if uu == u and vv == v), 0.0)  # find size of edge u->v
        return size_kB / self.bus_bw  # seconds

    def topo_order(self):
        """
        Topological order via Kahn's algorithm.
        Build indegree[] and adjacency g, then pop sources in FIFO order.
        """
        indeg = [0] * self.n_tasks                 # number of incoming edges per node
        g = defaultdict(list)                      # adjacency list (successors)

        for u, v, _ in self.edges:
            g[u].append(v)                         # add edge u->v
            indeg[v] += 1                          # increment indegree of v

        q = deque([i for i in range(self.n_tasks) if indeg[i] == 0])  # start from indegree-0 nodes
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            # for each successor of u, decrement indegree; if it becomes 0, enqueue it
            for w in g[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        return order

    def compute_t_b_levels(self, mapping):
        """
        Compute t-level and b-level under the given mapping.
        - t-level(u): earliest time inputs for u are ready (not including u's own runtime)
        - b-level(u): time from starting u to the DAG end (including u's runtime)
        Both depend on communication delays implied by the mapping.
        """
        # Execution time of each task on its assigned PE
        exec_times_map = np.array([self.exec_time[t, mapping[t]] for t in range(self.n_tasks)])

        # Build predecessor/successor lists with payload sizes
        # preds[i] = [(u, size_kB), ...] where u -> i
        # succs[i] = [(v, size_kB), ...] where i -> v
        preds = {i: [(u, s) for u, v, s in self.edges if v == i] for i in range(self.n_tasks)}
        succs = {i: [(v, s) for u, v, s in self.edges if u == i] for i in range(self.n_tasks)}
        order = self.topo_order()

        # t-level: forward pass
        tlevel = np.zeros(self.n_tasks)  # earliest start times for tasks
        for u in order:
            if preds[u]:
                tlevel[u] = max(tlevel[p] + exec_times_map[p] + self.comm_delay(mapping, p, u)
                                for p, _ in preds[u])
            else:
                tlevel[u] = 0.0

        # b-level: backward pass
        blevel = np.zeros(self.n_tasks)  # time-to-end from each task start
        for u in reversed(order):
            if succs[u]:
                blevel[u] = exec_times_map[u] + max(blevel[s] + self.comm_delay(mapping, u, s)
                                                    for s, _ in succs[u])
            else:
                blevel[u] = exec_times_map[u]

        return tlevel, blevel

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate one solution (mapping: task -> PE):
        - Build a static priority list by (t-level desc, b-level desc)
        - Do list scheduling (no bus contention), compute finish times
        - Objectives: Cost(=10*Energy), Energy (compute+comm), Makespan
        """
        mapping = np.asarray(np.rint(x), dtype=int)  # ensure integer mapping

        # Priority: bigger t-level first; tie-break by b-level
        tlevel, blevel = self.compute_t_b_levels(mapping)
        priority_list = sorted(range(self.n_tasks), key=lambda t: (-tlevel[t], -blevel[t]))

        pe_ready = np.zeros(self.n_pe)              # when each PE becomes free
        finish = np.zeros(self.n_tasks)             # finish time per task
        total_energy = 0.0
        total_comm_energy = 0.0

        preds = {i: [(u, s) for u, v, s in self.edges if v == i] for i in range(self.n_tasks)}

        # Static list scheduling
        for t in priority_list:
            pe = mapping[t]
            est = pe_ready[pe]                      # earliest start wrt PE availability

            # wait for all predecessors (plus comm delay if on different PE)
            for p, size_kB in preds[t]:
                pred_ready = finish[p]
                if mapping[p] != pe:
                    pred_ready += size_kB / self.bus_bw
                    total_comm_energy += self.bus_power * (size_kB / self.bus_bw)
                est = max(est, pred_ready)

            rt = self.exec_time[t, pe]              # runtime of task t on chosen PE
            st = est
            ft = st + rt

            finish[t] = ft
            pe_ready[pe] = ft
            total_energy += rt * self.pe_power[pe]  # compute energy (power * time)

        makespan = float(np.max(finish))
        energy = float(total_energy + total_comm_energy)
        cost = energy * 10.0                        # artificial third objective

        out["F"] = np.array([cost, energy, makespan], dtype=float)


# -----------------------
# Operators
# -----------------------
class MappingSampling(Sampling):
    """Sampling operator: generate random mappings (each task -> random PE)."""
    def _do(self, problem, n_samples, **kwargs):
        return np.random.randint(0, problem.n_pe, size=(n_samples, problem.n_tasks))

class MappingCrossover(Crossover):
    """Uniform crossover on integer genomes (task -> PE)."""
    def __init__(self, p=0.5):
        super().__init__(2, 2)  # 2 parents -> 2 offsprings
        self.p = p
    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape  # X: (n_parents, n_matings, n_var)
        Y = np.empty_like(X)
        for k in range(n_matings):
            a = np.rint(X[0, k]).astype(int)       # parent A
            b = np.rint(X[1, k]).astype(int)       # parent B
            mask = np.random.rand(n_var) < self.p  # per-gene choose parent A with prob p
            Y[0, k] = np.where(mask, a, b)         # child 1
            Y[1, k] = np.where(mask, b, a)         # child 2 (complement)
        return Y

class MappingMutation(Mutation):
    """Mutation: with prob p_gene per gene, reassign to a random PE."""
    def __init__(self, p_gene=0.2):
        super().__init__()
        self.p_gene = p_gene
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            xi = np.rint(X[i]).astype(int)
            mask = np.random.rand(problem.n_tasks) < self.p_gene
            if mask.any():
                xi[mask] = np.random.randint(0, problem.n_pe, size=np.sum(mask))
            X[i] = xi
        return X

class MappingDuplicate(ElementwiseDuplicateElimination):
    """Eliminate exact duplicate mappings (after rounding to int)."""
    def is_equal(self, a, b):
        return np.array_equal(np.rint(a.X).astype(int), np.rint(b.X).astype(int))


# -----------------------
# Callback: per-epoch summary & trends
# -----------------------
class VerboseTrace(Callback):
    """
    Prints a concise summary each generation and stores trends:
    - #ND solutions, min cost/energy/latency on the ND front
    - Hypervolume if available (pymoo.indicators.hv.HV)
    Also prints top-k ND solutions by best latency.
    """
    def __init__(self, top_k=3, print_every=1):
        super().__init__()
        self.top_k = top_k
        self.print_every = print_every

        # time series
        self.gens = []
        self.nd_sizes = []
        self.minC = []
        self.minE = []
        self.minL = []
        self.hv_vals = []

        # internal HV state
        self._hv = None
        self._ref_point = None

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        X = algorithm.pop.get("X")
        gen = algorithm.n_gen

        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        ND = F[nd_idx]

        # record series
        self.gens.append(gen)
        self.nd_sizes.append(len(nd_idx))
        if len(ND) > 0:
            self.minC.append(float(ND[:, 0].min()))
            self.minE.append(float(ND[:, 1].min()))
            self.minL.append(float(ND[:, 2].min()))
        else:
            self.minC.append(float("nan"))
            self.minE.append(float("nan"))
            self.minL.append(float("nan"))

        # init HV once with a ref point slightly worse than current population
        if HV is not None and self._hv is None:
            self._ref_point = F.max(axis=0) * 1.1
            self._hv = HV(ref_point=self._ref_point)

        hv_val = None
        if self._hv is not None and len(ND) > 0:
            hv_val = float(self._hv.do(ND))
            self.hv_vals.append(hv_val)

        # print console summary
        if gen % self.print_every == 0:
            hv_str = f"  HV={hv_val:.4f}" if hv_val is not None else ""
            print(f"[Gen {gen:02d}] ND={len(nd_idx):2d}  "
                  f"minC={self.minC[-1]:.4f}  minE={self.minE[-1]:.4f}  minL={self.minL[-1]:.4f}{hv_str}")

            if self.top_k > 0 and len(ND) > 0:
                # show a few best-by-latency solutions on the ND front
                order = np.argsort(ND[:, 2])[:self.top_k]
                for r, j in enumerate(order, 1):
                    i_pop = nd_idx[j]
                    f = ND[j]
                    x = np.rint(X[i_pop]).astype(int).tolist()
                    print(f"   - Top{r} by Lat: map={x}  F=[C={f[0]:.4f}, E={f[1]:.4f}, L={f[2]:.4f}]")


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

# Use VerboseTrace to see per-epoch (generation) output and keep trends
trace = VerboseTrace(top_k=3, print_every=1)

res = minimize(problem,
               algorithm,
               ('n_gen', 15),
               seed=1,
               callback=trace,
               verbose=True)   # set to False if you only want the custom prints


# -----------------------
# Print chromosomes (decision vectors) of final Pareto front
# -----------------------
X = res.X
F = res.F  # (n_points, 3) with [Cost, Energy, Latency]

nds = NonDominatedSorting()
nd_idx = nds.do(F, only_non_dominated_front=True)

# Ensure integer mapping even if operators passed floats / object dtype
if isinstance(X, np.ndarray) and X.dtype != object:
    X_int = np.rint(X).astype(int)
else:
    X_int = np.array([np.rint(np.asarray(xx)).astype(int) for xx in X])

print("\n=== Final Pareto Front Chromosomes (task -> PE) ===")
for k, i in enumerate(nd_idx):
    c, e, l = F[i]
    print(f"Sol#{k}: map={X_int[i].tolist()}   "
          f"F=[Cost={c:.4f}, Energy={e:.4f}, Latency={l:.4f}]")


# -----------------------
# Final Pareto front plot
# Show ALL solutions in gray; overlay Pareto-optimal in blue
# -----------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# all solutions (gray)
ax.scatter(F[:, 0], F[:, 1], F[:, 2],
           s=35, alpha=0.6, c='0.7',
           label=f"All solutions ({len(F)})",
           depthshade=False)

# non-dominated (blue)
ax.scatter(F[nd_idx, 0], F[nd_idx, 1], F[nd_idx, 2],
           s=65, alpha=0.95,
           label=f"Non-dominated ({len(nd_idx)})",
           depthshade=False)

# annotate ND as Sol#k (to match the printout above)
for k, i in enumerate(nd_idx):
    ax.text(F[i, 0], F[i, 1], F[i, 2], f"Sol#{k}", fontsize=9)

ax.set_title("Final Pareto Front (blue) over all solutions (gray)")
ax.set_xlabel("Cost")
ax.set_ylabel("Energy")
ax.set_zlabel("Latency")
ax.legend()
plt.tight_layout()
plt.show()


# -----------------------
# Trend plots (optional): ND size, best objective values, Hypervolume
# -----------------------
# ND set size over generations
plt.figure()
plt.plot(trace.gens, trace.nd_sizes, marker='o')
plt.xlabel("Generation")
plt.ylabel("# Non-dominated solutions")
plt.title("ND set size over generations")
plt.grid(True)
plt.show()

# Min objective values over generations (measured on the ND front)
plt.figure()
plt.plot(trace.gens, trace.minC, marker='o', label="min Cost")
plt.plot(trace.gens, trace.minE, marker='o', label="min Energy")
plt.plot(trace.gens, trace.minL, marker='o', label="min Latency")
plt.xlabel("Generation")
plt.ylabel("Best (min) value on ND front")
plt.title("Objective trends over generations")
plt.legend()
plt.grid(True)
plt.show()

# Hypervolume over generations (only if HV is available)
if len(trace.hv_vals) == len([g for g in trace.gens if g is not None]) and len(trace.hv_vals) > 0:
    plt.figure()
    plt.plot(trace.gens, trace.hv_vals, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume over generations")
    plt.grid(True)
    plt.show()
