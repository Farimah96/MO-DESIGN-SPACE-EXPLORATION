# -*- coding: utf-8 -*-
# Variable-Length Chromosome (VL) MPSoC DSE per Madsen-style with NSGA-II (pymoo)
# Genome = { alloc: [counts per PE type], bind: List[List[tasks per instance in order]] }
# Step-1 Crossover on alloc, Step-2 per-task binding inheritance + validate/repair.
# Scheduler = list scheduling with t/b-level, ECT over buses/bridges, buffers/local memory.
# Objectives = [cost, energy, memory_violation_words]

import math
import hashlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# from dataclasses import dataclass   # REMOVED (we now use plain classes)
from typing import Dict, List, Tuple

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# =========================
# Global Config (adjustable)
# =========================
RNG = np.random.default_rng(7)

# Processing Element (PE) types (order defines alloc vector indices)
GPP, ASIC, FPGA = 0, 1, 2
PE_TYPES = [GPP, ASIC, FPGA]

# Component caps (soft). Genome is variable-length; caps only bound random generation/mutations.
PE_MAX_TOTAL = 8
BUS_MAX, BRIDGE_MAX = 4, 3

# Unit costs
COST_PE      = {GPP: 100, ASIC: 200, FPGA: 300}
COST_BUS     = {0: 60, 1: 100}      # 0=std, 1=fast
COST_BRIDGE  = 80

# Energy
ENERGY_TASK_PER_TYPE = {GPP: 1.0, ASIC: 0.9, FPGA: 1.1}   # scale per exec unit
ENERGY_COMM_PER_BIT  = 1e-6

# Bus bandwidths (bits / time-unit)
BUS_BW = {0: 50e6, 1: 200e6}

# Buffers & local memory (in 32-bit words)
DEFAULT_INBUF_WORDS   = 4096
DEFAULT_OUTBUF_WORDS  = 4096
DEFAULT_LOCAL_MEM_WORDS = 1 << 16   # 65536 words

# Target total tasks after hyperperiod (as paper case)
TARGET_TASKS = 530


# =========================
# Application Model (DAG)
# =========================
class Task:
    def __init__(self, id: int, period: int, deadline: int,
                 wcet_per_type: Dict[int, int],
                 energy_per_type: Dict[int, float],
                 mem_words: int):
        self.id = id
        self.period = period
        self.deadline = deadline
        self.wcet_per_type = wcet_per_type
        self.energy_per_type = energy_per_type
        self.mem_words = mem_words

    def __repr__(self):
        return (f"Task(id={self.id}, period={self.period}, deadline={self.deadline}, "
                f"wcet_per_type={self.wcet_per_type}, energy_per_type={self.energy_per_type}, "
                f"mem_words={self.mem_words})")


class Edge:
    def __init__(self, src: int, dst: int, msg_bits: int):
        self.src = src
        self.dst = dst
        self.msg_bits = msg_bits

    def __repr__(self):
        return f"Edge(src={self.src}, dst={self.dst}, msg_bits={self.msg_bits})"


class AppModel:
    def __init__(self, tasks: Dict[int, Task], edges: List[Edge], dag: nx.DiGraph):
        self.tasks = tasks
        self.edges = edges
        self.dag = dag

    def __repr__(self):
        return f"AppModel(tasks=<{len(self.tasks)} tasks>, edges=<{len(self.edges)} edges>)"


def synthesize_smartphone_case() -> AppModel:  # a method for creating an artificial application
    """Synthesize base graph replicated to ~530 tasks."""
    base_nodes = 23
    G = nx.DiGraph()
    tasks: Dict[int, Task] = {}
    edges: List[Edge] = []

    for i in range(base_nodes):
        period = int(RNG.integers(10, 30))
        deadline = period
        wcet = {
            GPP:  int(RNG.integers(5, 20)),
            ASIC: int(RNG.integers(3, 15)),
            FPGA: int(RNG.integers(4, 18)),
        }
        energy = {t: wcet[t] * ENERGY_TASK_PER_TYPE[t] for t in PE_TYPES}
        memw = int(RNG.integers(256, 1024))
        tasks[i] = Task(id=i, period=period, deadline=deadline,
                        wcet_per_type=wcet, energy_per_type=energy, mem_words=memw)
        G.add_node(i)

    # Sparse DAG edges
    for i in range(base_nodes-1):
        for j in range(i+1, base_nodes):
            if RNG.random() < 0.12: # between each 2 nodes ceate an edge with prob = 12%
                bits = int(RNG.integers(4_000, 300_000))  # 4 Kb .. 300 Kb
                G.add_edge(i, j)
                edges.append(Edge(src=i, dst=j, msg_bits=bits))

    # Replicate until >= 530, then trim
    bigG = nx.DiGraph()
    big_tasks: Dict[int, Task] = {}
    big_edges: List[Edge] = []
    nid = 0
    copies = math.ceil(TARGET_TASKS / base_nodes)
    for _ in range(copies):
        mapping = {}
        for u in G.nodes():
            tu = tasks[u]
            new_task = Task(
                id=nid, period=tu.period, deadline=tu.deadline,
                wcet_per_type=tu.wcet_per_type.copy(),
                energy_per_type=tu.energy_per_type.copy(),
                mem_words=tu.mem_words
            )
            big_tasks[nid] = new_task
            bigG.add_node(nid)
            mapping[u] = nid
            nid += 1
        for (u, v) in G.edges():
            e = next(ed for ed in edges if ed.src == u and ed.dst == v)
            bigG.add_edge(mapping[u], mapping[v])
            big_edges.append(Edge(src=mapping[u], dst=mapping[v], msg_bits=e.msg_bits))

    # Trim to 530 exactly
    if len(big_tasks) > TARGET_TASKS:
        to_remove = len(big_tasks) - TARGET_TASKS
        rm_ids = sorted(big_tasks.keys())[-to_remove:]
        for rid in rm_ids:
            if bigG.has_node(rid):
                for p in list(bigG.predecessors(rid)):
                    if bigG.has_edge(p, rid): bigG.remove_edge(p, rid)
                for s in list(bigG.successors(rid)):
                    if bigG.has_edge(rid, s): bigG.remove_edge(rid, s)
                bigG.remove_node(rid)
                big_tasks.pop(rid, None)
        big_edges = [e for e in big_edges if e.src in big_tasks and e.dst in big_tasks]

    assert len(big_tasks) == TARGET_TASKS
    return AppModel(tasks=big_tasks, edges=big_edges, dag=bigG)


APP = synthesize_smartphone_case()


# =========================
# Helpers: VL genome <-> effective architecture
# =========================
def flatten_instances(alloc: List[int]) -> List[int]:
    """Given alloc counts per type [#GPP, #ASIC, #FPGA], return instance type list length=sum(alloc)."""
    types = []
    for t_idx, cnt in enumerate(alloc):
        types.extend([t_idx] * int(max(0, cnt)))
    return types

def build_assignment_from_bind(bind: List[List[int]], n_tasks: int) -> np.ndarray:
    """Map each task -> PE index (instance index) based on bind lists."""
    ASSIGN = np.zeros(n_tasks, dtype=int)
    for pe_idx, seq in enumerate(bind):
        for task in seq:
            if 0 <= task < n_tasks:
                ASSIGN[task] = pe_idx
    return ASSIGN

def normalize_bind(alloc: List[int], bind: List[List[int]], n_tasks: int) -> List[List[int]]:
    """Ensure bind has exactly sum(alloc) lists and covers each task exactly once (dedupe + fill missing)."""
    total_inst = sum(int(max(0, c)) for c in alloc)
    if total_inst <= 0:
        total_inst = 1
    bind = [list(seq) for seq in bind[:total_inst]] + [[] for _ in range(max(0, total_inst - len(bind)))]
    seen = set()
    for i in range(len(bind)):
        new_seq = []
        for t in bind[i]:
            if 0 <= t < n_tasks and t not in seen:
                new_seq.append(int(t))
                seen.add(int(t))
        bind[i] = new_seq
    missing = [t for t in range(n_tasks) if t not in seen]
    for t in missing:
        tgt = int(np.argmin([len(s) for s in bind])) if bind else 0
        if not bind:
            bind = [[]]
        bind[tgt].append(int(t))
    return bind

def genome_fingerprint(genome: dict) -> int:
    """Build a deterministic 64-bit seed from genome content to keep fabric stable across evaluations."""
    h = hashlib.md5()
    h.update(bytes([len(genome["alloc"])]))
    for x in genome["alloc"]:
        h.update(int(x).to_bytes(2, "little", signed=False))
    h.update(bytes([len(genome["bind"])]))
    for seq in genome["bind"]:
        h.update(bytes([len(seq) % 251]))
        for t in seq[:64]:
            h.update(int(t).to_bytes(2, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little", signed=False)

def arch_from_genome(genome: dict, n_pe: int):
    """Deterministically derive a bus/bridge fabric from genome so objectives are stable."""
    seed = genome_fingerprint(genome)
    rg = np.random.default_rng(seed)

    BUS_count = int(rg.integers(1, BUS_MAX+1))
    BUS_TYPES_used = int(rg.integers(1, len(COST_BUS)+1))
    BRIDGE_count = int(rg.integers(0, min(BRIDGE_MAX, max(0, BUS_count-1)) + 1))

    BUS_types = np.array(rg.integers(0, BUS_TYPES_used, size=BUS_count), dtype=int)

    # PE incidence to buses: at least one bus per PE
    BUS_PE = np.zeros((BUS_count, n_pe), dtype=int)
    for p in range(n_pe):
        BUS_PE[int(rg.integers(0, BUS_count)), p] = 1
    BUS_PE = np.maximum(BUS_PE, (rg.random((BUS_count, n_pe)) < 0.15).astype(int))

    # bridges (no self-loop)
    BRIDGES = np.zeros((BRIDGE_count, 2), dtype=int)
    for k in range(BRIDGE_count):
        if BUS_count >= 2:
            a, b = rg.choice(np.arange(BUS_count), size=2, replace=False)
            BRIDGES[k] = [int(a), int(b)]

    return BUS_types, BUS_PE, BRIDGES


# =========================
# Scheduling (list scheduling + ECT + buffer/memory)
# =========================
def build_bus_graph(BUS_count: int, BRIDGES: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    for b in range(BUS_count): G.add_node(b)
    for a, b in BRIDGES:
        if a != b and a < BUS_count and b < BUS_count:
            G.add_edge(int(a), int(b))
    return G

def shortest_bus_path(bus_graph: nx.Graph, src_bus: int, dst_bus: int):
    if src_bus == dst_bus: return [src_bus]
    try:
        return nx.shortest_path(bus_graph, src_bus, dst_bus)
    except nx.NetworkXNoPath:
        return None

def t_b_levels(app: AppModel, mapping_pe_type: Dict[int, int], msg_time_on_path) -> Tuple[Dict[int, float], Dict[int, float]]:
    G = app.dag
    t_level = {u: 0.0 for u in G.nodes()}
    topo = list(nx.topological_sort(G))
    for u in topo:
        max_in = 0.0
        for p in G.predecessors(u):
            mt = msg_time_on_path(p, u)
            max_in = max(max_in, t_level[p] + app.tasks[p].wcet_per_type[mapping_pe_type.get(p, GPP)] + mt)
        t_level[u] = max_in

    b_level = {u: 0.0 for u in G.nodes()}
    for u in reversed(topo):
        if G.out_degree(u) == 0:
            pe_type = mapping_pe_type.get(u, GPP)
            b_level[u] = app.tasks[u].deadline - app.tasks[u].wcet_per_type[pe_type]
        else:
            min_succ = float("inf")
            for v in G.successors(u):
                mt = msg_time_on_path(u, v)
                pe_type_u = mapping_pe_type.get(u, GPP)
                val = b_level[v] - (app.tasks[u].wcet_per_type[pe_type_u] + mt)
                min_succ = min(min_succ, val)
            b_level[u] = min_succ
    return t_level, b_level

def list_schedule(app: AppModel, PE_types_vec: List[int], BUS_types: np.ndarray, BUS_PE: np.ndarray,
                  BRIDGES: np.ndarray, bind: List[List[int]], verbose=False):
    """Static list scheduling with ECT; tie-breaking along per-PE order from bind lists."""
    PE_count = len(PE_types_vec)
    BUS_count = len(BUS_types)

    # bus params
    bus_bw = [BUS_BW[int(BUS_types[b])] if BUS_count>0 else 0.0 for b in range(BUS_count)]
    bus_graph = build_bus_graph(BUS_count, BRIDGES)

    # PE index for each task from bind
    n_tasks = len(app.tasks)
    ASSIGN = build_assignment_from_bind(bind, n_tasks)

    # map instance -> set of buses
    pe_on_buses = {p: set(np.where(BUS_PE[:, p] == 1)[0].tolist()) for p in range(PE_count)} if BUS_count>0 else {p:set() for p in range(PE_count)}

    def find_path_bw(p_src, p_dst):
        """Pick a bus path and bottleneck BW. Common bus preferred; else use shortest path over bridges."""
        if BUS_count == 0:
            return None, 0.0
        common = pe_on_buses[p_src].intersection(pe_on_buses[p_dst])
        if common:
            b = list(common)[0]
            return [b], bus_bw[b]
        best_path, best_bw = None, 0.0
        for bs in pe_on_buses[p_src]:
            for bd in pe_on_buses[p_dst]:
                path = shortest_bus_path(bus_graph, int(bs), int(bd))
                if path is not None:
                    bw = min([bus_bw[b] for b in path]) if path else 0.0
                    if bw > best_bw:
                        best_path, best_bw = path, bw
        return best_path, best_bw

    # Fast edge lookup
    edge_bits = {(e.src, e.dst): e.msg_bits for e in app.edges}

    def msg_time_on_path(u, v):
        if BUS_count == 0:
            return 1e6
        pu, pv = ASSIGN[u], ASSIGN[v]
        path, bw = find_path_bw(pu, pv)
        bits = edge_bits.get((u, v), 0)
        if bw <= 0:
            return 1e6
        return bits / bw

    # priorities (t/b-level)
    mapping_pe_type = {u: PE_types_vec[ASSIGN[u]] for u in app.dag.nodes()}
    t_level, b_level = t_b_levels(app, mapping_pe_type, msg_time_on_path)
    base_priority = {u: -0.5*t_level[u] + 0.5*b_level[u] for u in app.dag.nodes()}

    # tie-break by given within-PE order (earlier position => higher priority)
    pos_in_pe = {}
    for pe_idx, seq in enumerate(bind):
        for rank, t in enumerate(seq):
            if 0 <= t < n_tasks:
                pos_in_pe[t] = rank

    def pri(u):
        return (base_priority[u], -pos_in_pe.get(u, 0))

    indeg = {u: app.dag.in_degree(u) for u in app.dag.nodes()}
    ready = [u for u in app.dag.nodes() if indeg[u] == 0]
    ready.sort(key=lambda u: pri(u))

    pe_time = np.zeros(PE_count, dtype=float)
    bus_time = np.zeros(BUS_count, dtype=float)
    inbuf  = {p: DEFAULT_INBUF_WORDS for p in range(PE_count)}
    outbuf = {p: DEFAULT_OUTBUF_WORDS for p in range(PE_count)}
    local_mem_used = {p: 0 for p in range(PE_count)}
    local_mem_cap  = {p: DEFAULT_LOCAL_MEM_WORDS for p in range(PE_count)}

    mem_violation_words = 0
    energy_total = 0.0
    start_time, finish_time = {}, {}
    late_tasks: List[int] = []

    def schedule_comm(u, v, rcv_ready_time):
        nonlocal mem_violation_words, energy_total, bus_time
        pu, pv = ASSIGN[u], ASSIGN[v]
        path, bw = find_path_bw(pu, pv)
        bits = edge_bits.get((u, v), 0)
        if (path is None) or (bw <= 0):
            energy_total += bits * ENERGY_COMM_PER_BIT
            return 1e6
        ect = max(finish_time[u], rcv_ready_time)
        t = ect
        dur = bits / bw
        for b in path:
            t = max(t, bus_time[b])
        for b in path:
            bus_time[b] = t + dur
        if outbuf[pu] * 32 < bits:
            extra_words = math.ceil((bits - outbuf[pu]*32)/32.0)
            local_mem_used[pu] += extra_words
            if local_mem_used[pu] > local_mem_cap[pu]:
                mem_violation_words += (local_mem_used[pu] - local_mem_cap[pu])
                local_mem_used[pu] = local_mem_cap[pu]
        energy_total += bits * ENERGY_COMM_PER_BIT
        return t + dur

    # list scheduling
    Q = ready[:]
    while Q:
        u = Q.pop(0)
        p = ASSIGN[u]
        pe_type = PE_types_vec[p]
        est = 0.0
        for pre in app.dag.predecessors(u):
            est = max(est, schedule_comm(pre, u, pe_time[p]))
        est = max(est, pe_time[p])
        st = est
        et = st + app.tasks[u].wcet_per_type[pe_type]
        start_time[u] = st
        finish_time[u] = et
        pe_time[p] = et
        energy_total += app.tasks[u].energy_per_type[pe_type]
        local_mem_used[p] += app.tasks[u].mem_words
        if local_mem_used[p] > local_mem_cap[p]:
            mem_violation_words += (local_mem_used[p] - local_mem_cap[p])
            local_mem_used[p] = local_mem_cap[p]
        if et > app.tasks[u].deadline:
            late_tasks.append(u)
        for v in app.dag.successors(u):
            indeg[v] -= 1
            if indeg[v] == 0: Q.append(v)
        Q.sort(key=lambda x: pri(x))

    pe_has_late = set(ASSIGN[u] for u in late_tasks)
    safe_pes = [pp for pp in range(PE_count) if pp not in pe_has_late]
    makespan = max(finish_time.values()) if finish_time else 0.0

    return {
        "finish_time": finish_time,
        "makespan": makespan,
        "energy": energy_total,
        "mem_violation_words": mem_violation_words,
        "late_tasks": late_tasks,
        "safe_pes": safe_pes,
        "ASSIGN": ASSIGN
    }


# =========================
# Objectives
# =========================
def compute_cost(PE_types_vec: List[int], BUS_types: np.ndarray, BRIDGES: np.ndarray) -> float:
    c = 0.0
    for t in PE_types_vec:
        c += COST_PE[int(t)]
    for bt in BUS_types:
        c += COST_BUS[int(bt)]
    c += len(BRIDGES) * COST_BRIDGE
    return float(c)


# =========================
# VL Repair (keeps genome consistent)
# =========================
class VLRepair(Repair):
    def _do(self, problem, X, **kwargs):
        """
        Ensure alloc non-negative, at least one instance overall,
        and bind shape matches alloc and covers tasks.
        X shape can be (n,1) with object cells or (n,) of dicts (be robust).
        """
        n_tasks = len(APP.tasks)
        def fix_one(ind):
            g = ind if isinstance(ind, dict) else ind[0]
            alloc = list(np.maximum(np.array(g["alloc"], dtype=int), 0))
            if sum(alloc) == 0:
                alloc[GPP] = 1
            bind = [list(seq) for seq in g["bind"]]
            bind = normalize_bind(alloc, bind, n_tasks)
            return {"alloc": alloc, "bind": bind}

        if X.ndim == 1:   # fallback
            repaired = [fix_one(ind) for ind in X]
            return np.array([[g] for g in repaired], dtype=object)

        repaired = []
        for i in range(X.shape[0]):
            repaired.append(fix_one(X[i, 0]))
        return np.array([[g] for g in repaired], dtype=object)


# =========================
# VL Sampling
# =========================
class VLSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        """Generate random variable-length genomes. Return shape (n_samples, 1)."""
        X = []
        n_tasks = len(APP.tasks)
        for _ in range(n_samples):
            max_left = PE_MAX_TOTAL
            alloc = []
            for t in PE_TYPES[:-1]:
                cnt = int(RNG.integers(0, max_left+1))
                alloc.append(cnt)
                max_left -= cnt
            alloc.append(int(max_left if RNG.random()<0.5 else RNG.integers(0, max_left+1)))
            if sum(alloc) == 0:
                alloc[RNG.integers(0, len(alloc))] = 1
            total_inst = sum(alloc)
            tasks = np.arange(n_tasks); RNG.shuffle(tasks)
            chunks = np.array_split(tasks, max(1, total_inst))
            bind = [list(map(int, ch)) for ch in chunks]
            bind = normalize_bind(alloc, bind, n_tasks)
            X.append({"alloc": alloc, "bind": bind})
        return np.array([[g] for g in X], dtype=object)


# =========================
# VL Crossover (two-stage, with validate/repair)
# =========================
class VLCrossover(Crossover):
    def __init__(self, p_uniform=0.5):
        super().__init__(2, 2)
        self.p_uniform = p_uniform

    def _do(self, problem, X, **kwargs):
        n_matings = X.shape[1]
        off = np.empty((2, n_matings, 1), dtype=object)
        n_tasks = len(APP.tasks)

        def clamp_alloc(al):
            al = [max(0, int(x)) for x in al]
            if sum(al) == 0:
                al[RNG.integers(0, len(al))] = 1
            while sum(al) > PE_MAX_TOTAL:
                i = int(np.argmax(al))
                al[i] -= 1
            return al

        def task_map(bind, alloc):
            """Map: task -> (type, rank_within_type). Requires len(bind)==sum(alloc)."""
            inst_type_list = []
            for t, cnt in enumerate(alloc):
                inst_type_list += [t]*cnt
            assert len(bind) == len(inst_type_list), "Parent bind/alloc mismatch; ensure repair ran."
            m = {}
            for inst_idx, seq in enumerate(bind):
                t = inst_type_list[inst_idx]
                rank_same_type = sum(1 for j in range(inst_idx) if inst_type_list[j] == t)
                for task in seq:
                    if 0 <= task < n_tasks:
                        m[int(task)] = (t, rank_same_type)
            return m

        def inherit_bind(child_alloc, parentA_bind, parentB_bind, parentA_alloc, parentB_alloc):
            total_inst = sum(child_alloc)
            child_bind = [[] for _ in range(total_inst)]
            parentA_bind = normalize_bind(parentA_alloc, parentA_bind, n_tasks)
            parentB_bind = normalize_bind(parentB_alloc, parentB_bind, n_tasks)
            mapA = task_map(parentA_bind, parentA_alloc)
            mapB = task_map(parentB_bind, parentB_alloc)
            child_type_list = []
            for t, cnt in enumerate(child_alloc):
                child_type_list += [t]*cnt

            def place_task(task, target_type=None):
                if len(child_bind) == 0:
                    return
                cands = [i for i, tt in enumerate(child_type_list) if (target_type is None or tt==target_type)]
                if not cands:
                    cands = list(range(len(child_bind)))
                sizes = [len(child_bind[i]) for i in cands]
                i_sel = cands[int(np.argmin(sizes))]
                child_bind[i_sel].append(int(task))

            for task in range(n_tasks):
                inherit_from_A = (RNG.random() < 0.5)
                src = mapA if inherit_from_A else mapB
                alt = mapB if inherit_from_A else mapA
                placed = False
                t_info = src.get(task, None)
                if t_info is not None:
                    tt, _ = t_info
                    if child_alloc[tt] > 0:
                        place_task(task, target_type=tt)
                        placed = True
                if not placed:
                    t_info2 = alt.get(task, None)
                    if t_info2 is not None:
                        tt2, _ = t_info2
                        if child_alloc[tt2] > 0:
                            place_task(task, target_type=tt2)
                            placed = True
                if not placed:
                    tt3 = int(np.argmax(child_alloc)) if sum(child_alloc)>0 else None
                    place_task(task, target_type=tt3)

            return normalize_bind(child_alloc, child_bind, n_tasks)

        for k in range(n_matings):
            p1 = X[0, k, 0]
            p2 = X[1, k, 0]
            a1, b1 = list(p1["alloc"]), [list(s) for s in p1["bind"]]
            a2, b2 = list(p2["alloc"]), [list(s) for s in p2["bind"]]

            # Step 1: allocation crossover
            if RNG.random() < self.p_uniform:
                c_alloc1 = [a1[i] if RNG.random()<0.5 else a2[i] for i in range(len(PE_TYPES))]
                c_alloc2 = [a2[i] if RNG.random()<0.5 else a1[i] for i in range(len(PE_TYPES))]
            else:
                cx = int(RNG.integers(1, len(PE_TYPES)))
                c_alloc1 = a1[:cx] + a2[cx:]
                c_alloc2 = a2[:cx] + a1[cx:]
            c_alloc1 = clamp_alloc(c_alloc1)
            c_alloc2 = clamp_alloc(c_alloc2)

            # Step 2: per-task binding inheritance + repair
            c_bind1 = inherit_bind(c_alloc1, b1, b2, a1, a2)
            c_bind2 = inherit_bind(c_alloc2, b2, b1, a2, a1)

            off[0, k, 0] = {"alloc": c_alloc1, "bind": c_bind1}
            off[1, k, 0] = {"alloc": c_alloc2, "bind": c_bind2}

        return off


# =========================
# VL Mutation (paper-style operators) — FIXED safe index handling
# =========================
class VLMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        """
        X shape: (n,1) with object genomes.
        Operators: change_pe, add_pe, remove_pe, random_reassign_task, heuristic_reassign_task
        """
        n = len(X)
        Y = np.empty((n, 1), dtype=object)
        n_tasks = len(APP.tasks)

        for i in range(n):
            g = X[i, 0]
            alloc = list(g["alloc"])
            bind  = [list(s) for s in g["bind"]]
            total_inst = sum(alloc)

            action = RNG.choice(
                ["change_pe", "add_pe", "remove_pe",
                 "random_reassign_task", "heuristic_reassign_task"],
                p=[0.20, 0.20, 0.15, 0.25, 0.20]
            )

            if action == "change_pe" and total_inst>0:
                inst = int(RNG.integers(0, total_inst))
                t_list = flatten_instances(alloc)
                cur_t = t_list[inst]
                new_t = int(RNG.choice(PE_TYPES))
                if new_t != cur_t:
                    alloc[cur_t] = max(0, alloc[cur_t]-1)
                    alloc[new_t] += 1

            elif action == "add_pe" and sum(alloc) < PE_MAX_TOTAL:
                t = int(RNG.choice(PE_TYPES))
                alloc[t] += 1
                bind.append([])
                portion = max(1, len(APP.tasks)//max(1, sum(alloc))//2)
                all_tasks = [tt for seq in bind[:-1] for tt in seq]
                if all_tasks:
                    mv = RNG.choice(all_tasks, size=min(portion, len(all_tasks)), replace=False)
                    mv_set = set(int(x) for x in mv)
                    for seq in bind[:-1]:
                        seq[:] = [x for x in seq if x not in mv_set]
                    bind[-1].extend(int(x) for x in mv)

            elif action == "remove_pe" and total_inst > 1:
                inst = int(RNG.integers(0, total_inst))
                t_list = flatten_instances(alloc)
                cur_t = t_list[inst]
                alloc[cur_t] = max(0, alloc[cur_t]-1)
                gone_tasks = bind[inst]
                bind.pop(inst)
                if bind:  # redistribute if there are instances left
                    for tsk in gone_tasks:
                        dst = int(np.argmin([len(s) for s in bind]))
                        bind[dst].append(int(tsk))
                else:
                    # pathological: shouldn't happen because total_inst>1
                    bind = [list(gone_tasks)]

            elif action == "random_reassign_task" and total_inst>0:
                for _ in range(int(RNG.integers(1, 5))):
                    srcs = [ii for ii, s in enumerate(bind) if len(s)>0]
                    if not srcs: break
                    src = int(RNG.choice(srcs))
                    tsk = int(bind[src].pop(RNG.integers(0, len(bind[src]))))
                    dst = int(RNG.integers(0, len(bind)))
                    bind[dst].append(tsk)

            elif action == "heuristic_reassign_task" and total_inst>0:
                le = getattr(problem, "last_eval", None)
                if le is not None and le.get("late_tasks"):
                    tsk = int(RNG.choice(le["late_tasks"]))
                    # ensure tsk exists in some list; if not, skip removal step
                    present = False
                    for s in bind:
                        if tsk in s:
                            s.remove(tsk)
                            present = True
                            break
                    if not present:
                        # if not present (due to prior edits), it's fine; we will just place it
                        pass
                    safe = le.get("safe_pes", list(range(len(bind))))
                    # CLAMP SAFE INDICES to current bind length
                    safe = [ii for ii in safe if 0 <= ii < len(bind)]
                    if not safe:
                        safe = list(range(len(bind)))
                    sizes = [len(bind[ii]) for ii in safe]
                    dst = safe[int(np.argmin(sizes))]
                    bind[dst].append(tsk)

            # small jitter on alloc counts
            for j in range(len(alloc)):
                if RNG.random() < 0.15:
                    alloc[j] = max(0, alloc[j] + int(RNG.choice([-1, 1])))

            # normalize via repair to ensure consistency with new alloc
            repaired = VLRepair()._do(None, np.array([[{"alloc": alloc, "bind": bind}]], dtype=object))[0, 0]
            Y[i, 0] = repaired

        return Y


# =========================
# Problem
# =========================
class FlexMPSoC_VL(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=3, n_constr=0, elementwise_evaluation=True)
        self.last_eval = None

    def _evaluate(self, x, out, *args, **kwargs):
        """x is either dict or [dict]."""
        if not isinstance(x, dict):
            x = x[0]
        alloc = list(x["alloc"])
        bind  = [list(s) for s in x["bind"]]
        bind = normalize_bind(alloc, bind, len(APP.tasks))

        # Effective PE instance types according to alloc
        PE_types_vec = flatten_instances(alloc)

        # Deterministic bus/bridge fabric derived from genome to keep evaluation stable
        BUS_types, BUS_PE, BRIDGES = arch_from_genome({"alloc": alloc, "bind": bind}, len(PE_types_vec))

        sched = list_schedule(APP, PE_types_vec, BUS_types, BUS_PE, BRIDGES, bind, verbose=False)
        cost  = compute_cost(PE_types_vec, BUS_types, BRIDGES)
        energy = sched["energy"]
        memviol = sched["mem_violation_words"]

        # keep details for heuristic mutation
        self.last_eval = {
            "late_tasks": sched["late_tasks"],
            "safe_pes": sched["safe_pes"]
        }

        out["F"] = np.array([cost, energy, float(memviol)], dtype=float)


# =========================
# Logging (per generation)
# =========================
class ProgressPrinter:
    def __init__(self):
        self.header = False

    def __call__(self, algorithm):
        if not self.header:
            print("="*95)
            print("  n_gen |  n_eval | n_nds |         F_min(cost, energy, memV_words)         |  pop")
            print("="*95)
            self.header = True
        gen = algorithm.n_gen
        neval = algorithm.evaluator.n_eval
        F = algorithm.pop.get("F")
        ranks = algorithm.pop.get("rank")
        nnds = int(np.sum(ranks == 0))
        fmins = F.min(axis=0)
        print(f"{gen:7d} | {neval:7d} | {nnds:5d} | "
              f"{fmins[0]:10.2f}, {fmins[1]:10.2e}, {fmins[2]:9.0f} | {len(algorithm.pop):5d}")


# =========================
# Run NSGA-II
# =========================
if __name__ == "__main__":
    problem = FlexMPSoC_VL()
    algorithm = NSGA2(
        pop_size=80,
        sampling=VLSampling(),
        crossover=VLCrossover(),
        mutation=VLMutation(),
        repair=VLRepair(),
        eliminate_duplicates=False   # object genomes; equality not trivial
    )

    print("Starting NSGA-II (VL-Genome) ... (100 generations × 80 pop)")
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", 100),
        seed=2,
        save_history=False,
        verbose=False,
        callback=ProgressPrinter()
    )

    # Show Pareto-like sample (4 solutions sorted by cost)
    print("\nPareto solutions (sample):")
    print("idx |  price   energy           memV(words)")
    if len(res.F) > 0:
        order = np.argsort(res.F[:, 0])[:4]
        for i, k in enumerate(order):
            print(f"{i:3d} | {res.F[k,0]:7.0f}  {res.F[k,1]:.3e}   {res.F[k,2]:9.0f}")
    else:
        print("(no solutions)")

    # 3D scatter of non-dominated front
    F = res.F
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F_nd = F[nds]
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(F_nd[:,0], F_nd[:,1], F_nd[:,2], s=18)
    ax.set_xlabel('Cost')
    ax.set_ylabel('Energy')
    ax.set_zlabel('Memory Violation (words)')
    ax.set_title('Pareto Front (Non-dominated) - VL Chromosome')
    plt.tight_layout()
    plt.show()
