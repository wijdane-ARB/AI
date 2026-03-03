from grid import Grid
from astar import astar
from markov import build_transition_matrix, simulate_markov, compute_exact_absorption, analyze_markov_classes
from utils import plot_comparison_algorithms, plot_epsilon_impact, plot_grid

def create_grids():
    easy   = Grid(5, 5, obstacles=[(2,2)], start=(0,0), goal=(4,4))
    medium = Grid(5, 5, obstacles=[(1,1),(1,2),(3,2),(3,3)], start=(0,0), goal=(4,4))
    hard   = Grid(7, 7, obstacles=[(2,2),(2,3),(2,4),(4,2),(4,3),(4,4),(5,5),(3,5)], 
                  start=(0,0), goal=(6,6))
    return {'Facile': easy, 'Moyenne': medium, 'Difficile': hard}

def run_experiments():
    grids = create_grids()
    algo_results = []
    epsilons = [0.0, 0.1, 0.2, 0.3]
    epsilon_probs = []

    print("======================================================")
    print("   MINI-PROJET : A* + CHAÎNES DE MARKOV (Phases 1-5)")
    print("======================================================\n")

    for name, grid in grids.items():
        print(f"\n=== Grille {name} ===")
        
        path, cost, expanded = astar(grid, grid.start, grid.goal)
        print(f"A* (Manhattan)     : coût = {cost}, nœuds = {expanded}")
        
        _, ucs_cost, ucs_exp = astar(grid, grid.start, grid.goal, weight=0)
        print(f"UCS (weight=0)     : coût = {ucs_cost}, nœuds = {ucs_exp}")
        
        _, greedy_cost, greedy_exp = astar(grid, grid.start, grid.goal, pure_greedy=True)
        print(f"Greedy pur         : coût = {greedy_cost}, nœuds = {greedy_exp}")
        
        _, _, h0_exp = astar(grid, grid.start, grid.goal, use_h_zero=True)
        print(f"A* avec h=0        : nœuds = {h0_exp} (équiv. UCS)")

        P, states, idx = build_transition_matrix(grid, path, epsilon=0.2)
        start_idx = idx[grid.start]
        goal_idx  = idx[grid.goal]
        
        p_success, avg_time = simulate_markov(P, start_idx, goal_idx, n_sim=10000)
        print(f"Probabilité GOAL (simulation ε=0.2) : {p_success:.4f}  (temps moyen {avg_time:.1f} pas)")
        
        # APPEL CORRIGÉ (avec states et idx)
        p_exact = compute_exact_absorption(P, start_idx, goal_idx, states, idx)
        print(f"Probabilité EXACTE (matrice fondamentale) : {p_exact:.4f}")

        analyze_markov_classes(P, states, idx, goal_idx)

        algo_results.append({
            'name': name,
            'A_cost': cost,
            'A_nodes': expanded,
            'UCS_nodes': ucs_exp,
            'Greedy_nodes': greedy_exp
        })

        if name == 'Difficile':
            plot_grid(grid, path, trap_states=[(3,3), (3,4)], filename="hard_grid_with_trap.png")

        if name == 'Moyenne':
            probs = []
            for eps in epsilons:
                P_eps, _, _ = build_transition_matrix(grid, path, epsilon=eps)
                p, _ = simulate_markov(P_eps, start_idx, goal_idx, n_sim=5000)
                probs.append(p)
            epsilon_probs = probs

    plot_comparison_algorithms(algo_results)
    plot_epsilon_impact(epsilons, epsilon_probs)

    print("\n✅ Tout est terminé ! Fichiers PNG générés.")

if __name__ == "__main__":
    run_experiments()