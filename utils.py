import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle   

def plot_comparison_algorithms(results):
    names = [r['name'] for r in results]
    a_cost = [r['A_cost'] for r in results]
    a_nodes = [r['A_nodes'] for r in results]
    ucs_nodes = [r['UCS_nodes'] for r in results]
    greedy_nodes = [r['Greedy_nodes'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(names))
    ax1.bar(x, a_cost, color='skyblue', alpha=0.8, label='Coût (A*)')
    ax1.set_ylabel('Coût du chemin')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)

    ax2 = ax1.twinx()
    ax2.plot(x, a_nodes, 'o-', color='blue', label='A* nœuds')
    ax2.plot(x, ucs_nodes, 's-', color='red', label='UCS nœuds')
    ax2.plot(x, greedy_nodes, '^-', color='green', label='Greedy nœuds')
    ax2.set_ylabel('Nœuds développés')

    plt.title('Phase 2 — Comparaison UCS / Greedy / A*')
    fig.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('comparison_algorithms.png', dpi=300)
    plt.close()
    print("Sauvegardé : comparison_algorithms.png")


def plot_epsilon_impact(epsilons, probs):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilons, probs, marker='o', color='green', linewidth=2)
    plt.xlabel('ε (incertitude de glissement)')
    plt.ylabel('Probabilité d’atteindre GOAL')
    plt.title('Phase 5 — Robustesse face au glissement (grille Moyenne)')
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.grid(True)
    plt.savefig('epsilon_impact.png', dpi=300)
    plt.close()
    print("Sauvegardé : epsilon_impact.png")


def plot_grid(grid, path, trap_states=None, filename="grid_visualization.png"):
    fig, ax = plt.subplots(figsize=(grid.width * 0.6, grid.height * 0.6))
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_xticks(range(grid.width + 1))
    ax.set_yticks(range(grid.height + 1))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # Obstacles (noir)
    for (x, y) in grid.obstacles:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor='black'))

    # Pièges (rouge)
    if trap_states:
        for (x, y) in trap_states:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor='red', alpha=0.6))

    # Départ & But
    ax.add_patch(Rectangle((grid.start[0], grid.start[1]), 1, 1, facecolor='blue', alpha=0.7, label='Départ'))
    ax.add_patch(Rectangle((grid.goal[0], grid.goal[1]), 1, 1, facecolor='gold', alpha=0.9, label='But'))

    # Chemin A*
    if path:
        px = [p[0] + 0.5 for p in path]
        py = [p[1] + 0.5 for p in path]
        ax.plot(px, py, color='green', linewidth=3, marker='o', markersize=6, label='Chemin A*')

    ax.invert_yaxis()
    plt.title("Visualisation de la grille + pièges Markov")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"-> Visuel sauvegardé : {filename}")