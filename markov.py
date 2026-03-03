import numpy as np
import networkx as nx

def build_transition_matrix(grid, path, epsilon=0.1):
    policy = {}
    for i in range(len(path)-1):
        policy[path[i]] = path[i+1]

    states = grid.all_free_cells()
    if grid.goal not in states:
        states.append(grid.goal)
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))

    def local_heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    for s in states:
        i = state_to_idx[s]
        if s == grid.goal:
            P[i, i] = 1.0
            continue

        if s in policy:
            next_s = policy[s]
        else:
            neighbors = [(nx, ny) for nx, ny, _ in grid.neighbors(*s)]
            if neighbors:
                next_s = min(neighbors, key=lambda p: local_heuristic(p, grid.goal))
            else:
                next_s = s

        dx, dy = next_s[0] - s[0], next_s[1] - s[1]

        if (dx, dy) == (1, 0):   dirs = [(1,0), (0,1), (0,-1)]
        elif (dx, dy) == (-1,0): dirs = [(-1,0), (0,1), (0,-1)]
        elif (dx, dy) == (0,1):  dirs = [(0,1), (1,0), (-1,0)]
        elif (dx, dy) == (0,-1): dirs = [(0,-1), (1,0), (-1,0)]
        else:                    dirs = [(0,0)]

        probs = [1 - epsilon] + [epsilon/2] * (len(dirs)-1) if len(dirs) > 1 else [1.0]
        probs = probs[:len(dirs)]

        for d, p in zip(dirs, probs):
            tx, ty = s[0] + d[0], s[1] + d[1]
            target = (tx, ty)
            if grid.is_free(tx, ty) or target == grid.goal:
                j = state_to_idx.get(target, i)
                P[i, j] += p
            else:
                P[i, i] += p

    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
    return P, states, state_to_idx


def simulate_markov(P, start_idx, goal_idx, n_sim=10000, max_steps=200):
    successes = 0
    times = []
    for _ in range(n_sim):
        state = start_idx
        for t in range(max_steps):
            if state == goal_idx:
                successes += 1
                times.append(t)
                break
            state = np.random.choice(len(P), p=P[state])
    return successes / n_sim, np.mean(times) if times else float('inf')


def compute_exact_absorption(P, start_idx, goal_idx, states, state_to_idx):
    """Version ROBUSTE : gère les pièges multi-états comme [(3,3),(3,4)]"""
    n = P.shape[0]
    
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-8:
                G.add_edge(i, j)

    scc = list(nx.strongly_connected_components(G))
    
    recurrent_classes = []
    goal_class = None
    
    for component in scc:
        c_list = list(component)
        has_outgoing = False
        for i in c_list:
            for j in range(n):
                if j not in c_list and P[i, j] > 1e-8:
                    has_outgoing = True
                    break
            if has_outgoing: break
        if not has_outgoing:
            recurrent_classes.append(c_list)
            if goal_idx in c_list:
                goal_class = c_list

    if not goal_class:
        return 0.0

    all_recurrent = set()
    for cls in recurrent_classes:
        all_recurrent.update(cls)

    transient = [i for i in range(n) if i not in all_recurrent]

    if start_idx in all_recurrent:
        return 1.0 if start_idx in goal_class else 0.0

    if not transient:
        return 1.0 if start_idx in goal_class else 0.0

    # Construction Q et R
    num_trans = len(transient)
    num_abs = len(recurrent_classes)
    Q = np.zeros((num_trans, num_trans))
    R = np.zeros((num_trans, num_abs))
    trans_map = {old: new for new, old in enumerate(transient)}
    
    for i_idx, i in enumerate(transient):
        for j in range(n):
            prob = P[i, j]
            if prob < 1e-10: continue
            if j in transient:
                Q[i_idx, trans_map[j]] += prob
            else:
                abs_class_id = next(k for k, cls in enumerate(recurrent_classes) if j in cls)
                R[i_idx, abs_class_id] += prob

    I = np.eye(num_trans)
    try:
        N = np.linalg.inv(I - Q)
        B = N @ R
        start_trans_idx = trans_map[start_idx]
        goal_class_idx = recurrent_classes.index(goal_class)
        return B[start_trans_idx, goal_class_idx]
    except np.linalg.LinAlgError:
        print("Erreur : Matrice singulière.")
        return 0.0


def analyze_markov_classes(P, states, state_to_idx, goal_idx):
    G = nx.DiGraph()
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-8:
                G.add_edge(states[i], states[j])

    scc = list(nx.strongly_connected_components(G))
    print("=== Phase 4 — Classes de communication ===")
    print(f"Nombre de classes : {len(scc)}")
    for component in scc:
        c = list(component)
        outgoing = any(P[state_to_idx[s], state_to_idx[t]] > 1e-8 
                       for s in c for t in states if t not in c)
        if any(state_to_idx[s] == goal_idx for s in c):
            status = "RÉCURRENT ABSORBANT (GOAL)"
        elif not outgoing:
            status = "RÉCURRENT (PIÈGE)"
        else:
            status = "TRANSITOIRE"
        print(f"• {status} : {c}")