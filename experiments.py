"""
experiments.py — Pipeline principal : Phases 1 à 5 + Expériences E.1/E.2/E.3
==============================================================================
Orchestre toutes les expériences du mini-projet dans l'ordre des phases :

  Phase 1 : Définition des grilles (P1.1, P1.2, P1.3)
  Phase 2 : Planification A*, UCS, Greedy, comparaison (P2.1, P2.2, P2.3)
  Phase 3 : Construction de P, calcul π^(n) (P3.1, P3.2, P3.3)
  Phase 4 : Classes de communication, absorption exacte (P4.1, P4.2, P4.3)
  Phase 5 : Simulation Monte-Carlo, histogrammes, taux d'échec (P5.1, P5.2, P5.3)

  E.1 : comparaison UCS / Greedy / A* sur 3 grilles
  E.2 : variation de ε ∈ {0, 0.1, 0.2, 0.3} → P(GOAL), P(FAIL)
  E.3 : comparaison heuristique Manhattan vs h = 0
"""

from grid import Grid
from astar import astar
from markov import (build_transition_matrix, simulate_markov,
                    compute_exact_absorption, analyze_markov_classes,
                    compute_pi_n)
from utils import (plot_comparison_algorithms, plot_epsilon_impact,
                   plot_grid, plot_hitting_time_distribution,
                   plot_fail_rate_epsilon, plot_pi_n)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — Définition des grilles de test (P1.1, P1.2)
# ─────────────────────────────────────────────────────────────────────────────

def create_grids():
    """
    Crée trois grilles de difficulté croissante (P1.1).

    - Facile   : 5×5, 1 obstacle, chemin presque direct
    - Moyenne  : 5×5, 4 obstacles, nécessite un contournement
    - Difficile: 7×7, 8 obstacles, labyrinthe complexe avec zone FAIL

    Coût uniforme c = 1 pour tous les déplacements (P1.2).
    """
    # Grille 5×5 : un seul obstacle au centre
    easy = Grid(5, 5,
                obstacles=[(2, 2)],
                start=(0, 0), goal=(4, 4))

    # Grille 5×5 : quatre obstacles formant un couloir
    medium = Grid(5, 5,
                  obstacles=[(1, 1), (1, 2), (3, 2), (3, 3)],
                  start=(0, 0), goal=(4, 4))

    # Grille 7×7 : huit obstacles créant un chemin sinueux
    # Les états (3,3) et (3,4) seront identifiés comme pièges FAIL
    hard = Grid(7, 7,
                obstacles=[(2, 2), (2, 3), (2, 4),
                           (4, 2), (4, 3), (4, 4),
                           (5, 5), (3, 5)],
                start=(0, 0), goal=(6, 6))

    return {'Facile': easy, 'Moyenne': medium, 'Difficile': hard}


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments():
    """
    Exécute toutes les phases du mini-projet et génère les figures.
    """
    # Créer les grilles (Phase 1)
    grids = create_grids()

    # Conteneurs de résultats pour les figures de synthèse
    algo_results       = []     # résultats Phase 2 (comparaison algorithmes)
    epsilons           = [0.0, 0.1, 0.2, 0.3]   # valeurs de ε pour E.2 (P1.3)
    epsilon_probs      = []     # P(GOAL) pour chaque ε
    epsilon_fail_rates = []     # P(FAIL) pour chaque ε
    pi_n_results       = {}     # courbes π^(n) par grille

    print("=" * 60)
    print("   MINI-PROJET : A* + CHAÎNES DE MARKOV (Phases 1-5)")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    #  Boucle sur les trois grilles
    # ─────────────────────────────────────────────────────────────────────────
    for name, grid in grids.items():
        print(f"\n{'=' * 20} Grille {name} {'=' * 20}")

        # ── Phase 2 : Planification déterministe (P2.1, P2.2, P2.3) ──────────
        print("\n--- Phase 2 : Comparaison des algorithmes (E.1, E.3) ---")

        # A* standard avec heuristique Manhattan (P2.1)
        path, cost, exp, t_ms, open_max = astar(grid, grid.start, grid.goal)

        # UCS : weight = 0, f = g seulement (P2.2)
        _, uc, ue, ut, uo = astar(grid, grid.start, grid.goal, weight=0)

        # Greedy Best-First : f = h seulement (P2.2)
        _, gc, ge, gt, go = astar(grid, grid.start, grid.goal, pure_greedy=True)

        # A* avec h = 0 : équivalent UCS, mesure l'apport de l'heuristique (E.3)
        _, _, h0e, _, _ = astar(grid, grid.start, grid.goal, use_h_zero=True)

        # Afficher le tableau de comparaison (P2.3 : métriques)
        print(f"{'Algo':<22} {'Coût':>5} {'Nœuds':>7} {'Temps(ms)':>10} {'OPEN max':>9}")
        print(f"{'-' * 55}")
        print(f"{'A* (Manhattan)':<22} {cost:>5} {exp:>7} {t_ms:>10.3f} {open_max:>9}")
        print(f"{'UCS (weight=0)':<22} {uc:>5} {ue:>7} {ut:>10.3f} {uo:>9}")
        print(f"{'Greedy pur':<22} {gc:>5} {ge:>7} {gt:>10.3f} {go:>9}")
        print(f"{'A* avec h=0 (E.3)':<22} {'--':>5} {h0e:>7} {'--':>10} {'--':>9}")

        # Stocker pour la figure de comparaison (E.1)
        algo_results.append({
            'name': name,
            'A_cost': cost,   'A_nodes': exp,
            'UCS_nodes': ue,  'Greedy_nodes': ge
        })

        # ── Phase 3 : Construction de P + calcul π^(n) (P3.1, P3.2, P3.3) ───
        print("\n--- Phase 3 : Chaîne de Markov (ε = 0.2) ---")

        # Construire la matrice stochastique P à partir du chemin A* (P3.1)
        P, states, idx = build_transition_matrix(grid, path, epsilon=0.2)

        # Vérification que P est bien stochastique (somme lignes = 1) (P3.2)
        import numpy as np
        stoch_ok = np.allclose(P.sum(axis=1), 1.0)
        print(f"Matrice P : {P.shape[0]} états, stochastique = {stoch_ok}")

        # Indices des états clés dans la matrice P
        start_idx = idx[grid.start]
        goal_idx  = idx[grid.goal]

        # P3.3 : calculer π^(n) = π^(0) · P^n par récurrence
        pi_series  = compute_pi_n(P, start_idx, n_steps=30)
        goal_probs = [pi[goal_idx] for pi in pi_series]
        pi_n_results[name] = goal_probs

        # Afficher P(X_n = GOAL) à quelques instants clés
        print("π^(n)[GOAL] pour n = 5, 10, 15, 20, 25, 30 :")
        for step in [5, 10, 15, 20, 25, 30]:
            print(f"  n={step:2d} → {goal_probs[step - 1]:.4f}")

        # ── Phase 4 : Analyse des classes de communication (P4.1, P4.2, P4.3) ─
        print("\n--- Phase 4 : Classes de communication ---")

        # Identifier et afficher les classes (transitoires / récurrentes / FAIL)
        analyze_markov_classes(P, states, idx, goal_idx)

        # Calculer les probabilités d'absorption exactes par matrice fondamentale
        # N = (I - Q)^{-1},  B = N · R  (P4.3)
        p_exact_goal, p_exact_fail = compute_exact_absorption(
            P, start_idx, goal_idx, states, idx)
        print(f"Probabilité exacte GOAL : {p_exact_goal:.4f}")
        print(f"Probabilité exacte FAIL : {p_exact_fail:.4f}")

        # ── Phase 5 : Simulation Monte-Carlo (P5.1, P5.2, P5.3) ──────────────
        print("\n--- Phase 5 : Simulation Monte-Carlo (N = 10 000) ---")

        # Lancer 10 000 trajectoires et collecter les temps d'atteinte
        p_sim, avg_time, times_success, times_fail = simulate_markov(
            P, start_idx, goal_idx, n_sim=10000)

        # Calculer le taux d'échec empirique (P5.2)
        p_fail_sim = len(times_fail) / 10000

        print(f"P(GOAL) simulation  : {p_sim:.4f}   (temps moyen = {avg_time:.1f} pas)")
        if times_fail:
            print(f"P(FAIL) simulation  : {p_fail_sim:.4f}   "
                  f"(temps moyen avant FAIL = {np.mean(times_fail):.1f} pas)")
        else:
            print(f"P(FAIL) simulation  : {p_fail_sim:.4f}")

        # P5.3 : comparer simulation vs calcul exact
        print(f"Taux d'échec (simulation) : {p_fail_sim:.4f}")
        print(f"Écart |P_sim - P_exact|   : {abs(p_sim - p_exact_goal):.4f}")

        # ── Figures spécifiques à la grille Difficile ─────────────────────────
        if name == 'Difficile':

            # Histogrammes des temps d'atteinte GOAL et FAIL (P5.2)
            plot_hitting_time_distribution(
                times_success, times_fail, epsilon=0.2,
                filename='hitting_time_distribution.png')

            # Visualisation de la grille avec le chemin A* et les pièges FAIL
            plot_grid(grid, path,
                      trap_states=[(3, 3), (3, 4)],
                      filename='hard_grid_with_trap.png')

            # ── Expérience E.2 : variation de ε ──────────────────────────────
            print("\n--- E.2 : Variation de ε ∈ {0.0, 0.1, 0.2, 0.3} ---")
            for eps in epsilons:
                # Reconstruire P avec le nouvel ε
                P_e, _, idx_e = build_transition_matrix(grid, path, epsilon=eps)
                si = idx_e[grid.start]
                gi = idx_e[grid.goal]

                # Simuler et mesurer P(GOAL) et P(FAIL)
                p_e, _, _, ft = simulate_markov(P_e, si, gi, n_sim=10000)
                fr = len(ft) / 10000   # taux d'échec pour cet ε

                epsilon_probs.append(p_e)
                epsilon_fail_rates.append(fr)
                print(f"  ε={eps:.1f} → P(GOAL) = {p_e:.4f}  |  P(FAIL) = {fr:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Génération des figures de synthèse
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Génération des figures ---")

    # Figure 1 : comparaison UCS / Greedy / A* (E.1)
    plot_comparison_algorithms(algo_results)

    # Figure 2 : P(GOAL) en fonction de ε (E.2)
    plot_epsilon_impact(epsilons, epsilon_probs)

    # Figure 3 : P(FAIL) en fonction de ε
    plot_fail_rate_epsilon(epsilons, epsilon_fail_rates)

    # Figure 4 : évolution de π^(n)[GOAL] (P3.3)
    plot_pi_n(pi_n_results)

    print("\n Toutes les figures générées. Projet terminé !")


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiments()