"""
utils.py — Visualisations et figures pour les Phases 2, 3 et 5
================================================================
Contient toutes les fonctions de tracé matplotlib utilisées
pour illustrer les résultats du mini-projet :

  - Phase 2 : comparaison UCS / Greedy / A* (coût + nœuds)
  - Phase 3 : évolution de π^(n)[GOAL] = π^(0) · P^n
  - Phase 5 : histogrammes des temps d'atteinte, taux d'échec, ε-impact
  - Grille  : visualisation des obstacles, chemin A*, états FAIL
"""

# Forcer le backend non-interactif pour la génération de fichiers PNG
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — Comparaison des algorithmes de recherche
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_algorithms(results, filename='comparison_algorithms.png'):
    """
    Figure Phase 2 (P2.3) : comparaison UCS / Greedy / A* sur 3 grilles.

    Axe gauche  : coût du chemin optimal (barres bleues, A*).
    Axe droit   : nombre de nœuds développés pour chaque algorithme.

    Paramètres
    ----------
    results  : liste de dicts {'name', 'A_cost', 'A_nodes', 'UCS_nodes', 'Greedy_nodes'}
    filename : nom du fichier de sortie
    """
    # Extraire les données par algorithme
    names        = [r['name']         for r in results]
    a_cost       = [r['A_cost']       for r in results]
    a_nodes      = [r['A_nodes']      for r in results]
    ucs_nodes    = [r['UCS_nodes']    for r in results]
    greedy_nodes = [r['Greedy_nodes'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = list(range(len(names)))

    # ── Axe gauche : coût du chemin (barres) ─────────────────────────────────
    ax1.bar(x, a_cost, color='skyblue', alpha=0.8, label='Coût (A*)', width=0.5)
    ax1.set_ylabel('Coût du chemin', fontsize=12)
    ax1.set_ylim(0, max(a_cost) * 1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=12)

    # ── Axe droit : nombre de nœuds développés (lignes) ──────────────────────
    ax2 = ax1.twinx()
    ax2.plot(x, a_nodes,      'o-', color='blue',  label='A* nœuds',     linewidth=2, markersize=8)
    ax2.plot(x, ucs_nodes,    's-', color='red',   label='UCS nœuds',    linewidth=2, markersize=8)
    ax2.plot(x, greedy_nodes, '^-', color='green', label='Greedy nœuds', linewidth=2, markersize=8)
    ax2.set_ylabel('Nœuds développés', fontsize=12)
    ax2.set_ylim(0, max(max(a_nodes), max(ucs_nodes), max(greedy_nodes)) * 1.4)

    # Annoter chaque point avec sa valeur numérique
    for i, (an, un, gn) in enumerate(zip(a_nodes, ucs_nodes, greedy_nodes)):
        ax2.annotate(str(an), (x[i], an), textcoords="offset points", xytext=( 6,  4), color='blue',  fontsize=9)
        ax2.annotate(str(un), (x[i], un), textcoords="offset points", xytext=( 6, -14), color='red',   fontsize=9)
        ax2.annotate(str(gn), (x[i], gn), textcoords="offset points", xytext=( 6,  4), color='green', fontsize=9)

    # Fusionner les légendes des deux axes
    plt.title('Phase 2 — Comparaison UCS / Greedy / A*', fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Robustesse P(GOAL) en fonction de ε
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon_impact(epsilons, probs, filename='epsilon_impact.png'):
    """
    Figure Phase 5 (E.2) : P(GOAL) simulée en fonction de ε.

    Montre comment la probabilité d'atteindre le but diminue
    lorsque l'incertitude de glissement ε augmente.

    Paramètres
    ----------
    epsilons : liste des valeurs de ε testées
    probs    : liste des P(GOAL) correspondantes
    """
    plt.figure(figsize=(8, 5))

    # Tracer la courbe P(GOAL) vs ε
    plt.plot(epsilons, probs, marker='o', color='green', linewidth=2, markersize=8)

    # Annoter chaque point avec sa valeur exacte
    for eps, p in zip(epsilons, probs):
        plt.annotate(f'{p:.4f}', (eps, p),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, color='darkgreen')

    plt.xlabel('ε (incertitude de glissement)', fontsize=11)
    plt.ylabel("Probabilité d'atteindre GOAL", fontsize=11)
    plt.title('Phase 5 — Robustesse face au glissement (grille Difficile)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Distribution des temps d'atteinte (P5.2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hitting_time_distribution(times_success, times_fail, epsilon,
                                   filename='hitting_time_distribution.png'):
    """
    Figure P5.2 : histogrammes des temps d'atteinte GOAL et FAIL.

    Panneau gauche  : distribution du temps avant absorption en GOAL.
    Panneau droit   : distribution du temps avant absorption dans un piège FAIL.
    Répond à l'exigence 'distribution du temps d'atteinte' du cahier des charges.

    Paramètres
    ----------
    times_success : liste des temps (en pas) des trajectoires ayant atteint GOAL
    times_fail    : liste des temps (en pas) des trajectoires tombées dans un piège
    epsilon       : valeur de ε utilisée (pour le titre)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panneau gauche : temps d'atteinte de GOAL ─────────────────────────────
    if times_success:
        # Histogramme des temps de succès
        axes[0].hist(times_success,
                     bins=range(1, min(max(times_success) + 2, 80)),
                     color='steelblue', edgecolor='white', alpha=0.85)
        # Ligne verticale indiquant la moyenne
        mean_suc = np.mean(times_success)
        axes[0].axvline(mean_suc, color='red', linestyle='--', linewidth=2,
                        label=f'Moyenne = {mean_suc:.1f} pas')
        axes[0].legend(fontsize=10)

    axes[0].set_xlabel("Temps d'atteinte (pas)", fontsize=11)
    axes[0].set_ylabel('Fréquence', fontsize=11)
    axes[0].set_title(f"Distribution du temps d'atteinte — GOAL\n"
                      f"(n={len(times_success)} trajectoires réussies, ε={epsilon})",
                      fontsize=10)
    axes[0].set_xlim(0, 70)
    axes[0].grid(True, alpha=0.4)

    # ── Panneau droit : temps d'absorption dans un piège FAIL ─────────────────
    if times_fail:
        # Histogramme des temps d'échec
        axes[1].hist(times_fail,
                     bins=range(1, min(max(times_fail) + 2, 40)),
                     color='tomato', edgecolor='white', alpha=0.85)
        # Ligne verticale indiquant la moyenne
        mean_fail = np.mean(times_fail)
        axes[1].axvline(mean_fail, color='darkred', linestyle='--', linewidth=2,
                        label=f'Moyenne = {mean_fail:.1f} pas')
        axes[1].legend(fontsize=10)

    axes[1].set_xlabel("Temps avant absorption (pas)", fontsize=11)
    axes[1].set_ylabel('Fréquence', fontsize=11)
    axes[1].set_title(f"Distribution du temps d'absorption — FAIL\n"
                      f"(n={len(times_fail)} trajectoires échouées, ε={epsilon})",
                      fontsize=10)
    axes[1].grid(True, alpha=0.4)

    # Titre général de la figure
    plt.suptitle(f"Phase 5 (P5.2) — Distributions des temps d'atteinte "
                 f"(ε={epsilon}, grille Difficile)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sauvegardé : {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Taux d'échec P(FAIL) en fonction de ε
# ─────────────────────────────────────────────────────────────────────────────

def plot_fail_rate_epsilon(epsilons, fail_rates, filename='fail_rate_epsilon.png'):
    """
    Figure complémentaire (E.2) : taux d'échec P(FAIL) en fonction de ε.

    Complète plot_epsilon_impact() en montrant le risque de capture
    par un état piège (FAIL) pour chaque niveau d'incertitude.

    Paramètres
    ----------
    epsilons   : liste des valeurs de ε
    fail_rates : liste des P(FAIL) = 1 - P(GOAL)
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Couleurs progressives pour souligner l'augmentation du risque
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']

    # Tracer les barres P(FAIL) par ε
    bars = ax.bar([str(e) for e in epsilons], fail_rates,
                  color=colors[:len(epsilons)], edgecolor='white', width=0.5)

    # Annoter chaque barre avec sa valeur numérique
    for bar, val in zip(bars, fail_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('ε (taux de glissement)', fontsize=11)
    ax.set_ylabel("Taux d'échec P(FAIL)", fontsize=11)
    ax.set_title("Taux d'échec en fonction de ε (grille Difficile)", fontsize=12)
    ax.set_ylim(0, max(fail_rates) * 1.6 if max(fail_rates) > 0 else 0.1)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Évolution de π^(n) = π^(0) · P^n
# ─────────────────────────────────────────────────────────────────────────────

def plot_pi_n(results_per_grid, filename='pi_n_evolution.png'):
    """
    Figure Phase 3 (P3.3) : évolution de P(X_n = GOAL) au fil du temps.

    Montre la convergence de π^(n)[GOAL] vers 1 (état absorbant)
    pour chaque grille, en utilisant la récurrence π^(n) = π^(n-1) · P.

    Paramètres
    ----------
    results_per_grid : dict {nom_grille: [P(GOAL) pour n=1..N]}
    """
    plt.figure(figsize=(9, 5))

    # Une couleur par grille pour distinguer les courbes
    colors = {'Facile': 'green', 'Moyenne': 'orange', 'Difficile': 'red'}

    for name, goal_probs in results_per_grid.items():
        steps = list(range(1, len(goal_probs) + 1))
        plt.plot(steps, goal_probs,
                 marker='o', markersize=4,
                 color=colors.get(name, 'blue'),
                 label=name, linewidth=2)

    # Ligne de référence : probabilité maximale (état absorbant → 1.0)
    plt.axhline(1.0, color='gray', linestyle=':', linewidth=1)

    plt.xlabel('Pas n', fontsize=11)
    plt.ylabel('P(X_n = GOAL)', fontsize=11)
    plt.title("Phase 3 (P3.3) — Évolution de π^(n) = π^(0) · P^n", fontsize=12)
    plt.legend(fontsize=10)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Sauvegardé : {filename}")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation de la grille
# ─────────────────────────────────────────────────────────────────────────────

def plot_grid(grid, path, trap_states=None, filename='grid_visualization.png'):
    """
    Visualisation de la grille 2D avec obstacles, chemin A* et états FAIL.

    Code couleur :
      Noir  : obstacle (cellule bloquée)
      Rouge : état piège FAIL (classe récurrente non-GOAL)
      Bleu  : état initial s0 (Départ)
      Or    : état but GOAL
      Vert  : chemin A* optimal

    Paramètres
    ----------
    grid        : objet Grid
    path        : chemin A* (liste d'états)
    trap_states : liste des états pièges à colorier en rouge
    filename    : nom du fichier de sortie
    """
    fig, ax = plt.subplots(figsize=(grid.width * 1.0, grid.height * 1.0))
    W, H = grid.width, grid.height

    # Configurer les axes et la grille visuelle
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    def to_plot(col, row):
        """
        Convertit les coordonnées de grille (col, row) en coordonnées matplotlib.
        La ligne 0 de la grille est affichée en haut → inverser l'axe y.
        """
        return col, H - 1 - row

    # ── Obstacles : cellules noires ───────────────────────────────────────────
    for (col, row) in grid.obstacles:
        px, py = to_plot(col, row)
        ax.add_patch(Rectangle((px, py), 1, 1, facecolor='black'))

    # ── États pièges FAIL : cellules rouges ───────────────────────────────────
    if trap_states:
        for (col, row) in trap_states:
            px, py = to_plot(col, row)
            ax.add_patch(Rectangle((px, py), 1, 1, facecolor='red', alpha=0.6))

    # ── État initial s0 : cellule bleue ───────────────────────────────────────
    sc, sr = grid.start
    sx, sy = to_plot(sc, sr)
    ax.add_patch(Rectangle((sx, sy), 1, 1, facecolor='blue', alpha=0.7))

    # ── État but GOAL : cellule dorée ─────────────────────────────────────────
    gc, gr = grid.goal
    gx, gy = to_plot(gc, gr)
    ax.add_patch(Rectangle((gx, gy), 1, 1, facecolor='gold', alpha=0.9))

    # ── Chemin A* : ligne verte reliant les étapes ────────────────────────────
    if path:
        # Centrer chaque marqueur dans la cellule (+0.5)
        px_list = [to_plot(p[0], p[1])[0] + 0.5 for p in path]
        py_list = [to_plot(p[0], p[1])[1] + 0.5 for p in path]
        ax.plot(px_list, py_list, color='green', linewidth=3,
                marker='o', markersize=6)

    # ── Légende ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color='blue',  alpha=0.7, label='Départ'),
        mpatches.Patch(color='gold',  alpha=0.9, label='But (GOAL)'),
        mpatches.Patch(color='red',   alpha=0.6, label='Piège (FAIL)'),
        mpatches.Patch(color='black',            label='Obstacle'),
        plt.Line2D([0], [0], color='green', linewidth=2, marker='o', label='Chemin A*'),
    ]
    ax.legend(handles=legend_patches,
              loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=9)

    plt.title("Grille difficile : chemin A* et états FAIL (pièges Markov)", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"-> Visuel sauvegardé : {filename}")