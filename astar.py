"""
astar.py — Phase 2 : Planification déterministe avec A* et variantes
=======================================================================
Implémente A*, UCS, Greedy Best-First et Weighted A* sur une grille 2D.

Formule générale : f(n) = g(n) + w * h(n)
  - A* standard       : w = 1,  pure_greedy = False
  - UCS               : w = 0   (pas d'heuristique, exploration uniforme)
  - Greedy Best-First : pure_greedy = True  (f = h seulement)
  - Weighted A*       : w > 1   (compromis vitesse / optimalité)
  - A* avec h = 0     : use_h_zero = True  (équivalent UCS, expérience E.3)
"""

import heapq
import time


# ─────────────────────────────────────────────────────────────────────────────
#  Heuristiques
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_manhattan(a, b):
    """
    Distance de Manhattan — heuristique admissible et cohérente.

    Admissible  : h(n) <= h*(n) car chaque déplacement coûte exactement 1.
    Cohérente   : h(n) <= c(n, n') + h(n') pour tout successeur n'.
    → Garantit qu'A* trouve le chemin optimal sans réouverture de noeuds.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_zero(a, b):
    """
    Heuristique nulle h(n) = 0 (expérience E.3).
    A* dégénère en UCS : f = g uniquement.
    Admissible mais non informative → expansion identique à UCS.
    Permet de mesurer l'apport de l'heuristique Manhattan.
    """
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Algorithme A* générique
# ─────────────────────────────────────────────────────────────────────────────

def astar(grid, start, goal, weight=1.0, pure_greedy=False, use_h_zero=False):
    """
    Algorithme A* générique sur grille 2D (P2.1, P2.2, P2.3).

    Paramètres
    ----------
    grid        : objet Grid fournissant neighbors() et les coûts
    start       : état initial (x, y)
    goal        : état but    (x, y)
    weight      : facteur w pour Weighted A*  (défaut 1.0 = A* standard)
    pure_greedy : si True → f = h   (Greedy Best-First Search)
    use_h_zero  : si True → h = 0   (A* équivalent à UCS)

    Retourne
    --------
    (chemin, coût_total, noeuds_développés, temps_ms, taille_OPEN_max)
    """

    # ── Démarrage du chronomètre pour mesure de performance (P2.3) ───────────
    t_start = time.perf_counter()

    # ── OPEN : file de priorité min-heap, triée par f(n) ─────────────────────
    # Format : (f(n), état) — heapq donne toujours le minimum en tête
    open_set = []
    heapq.heappush(open_set, (0, start))

    # ── came_from : dictionnaire prédécesseur pour reconstruire le chemin ─────
    came_from = {}

    # ── g_score : coût minimal connu pour atteindre chaque état depuis start ──
    g_score = {start: 0}

    # ── CLOSED : ensemble des états définitivement développés ─────────────────
    closed = set()

    # ── Compteurs de métriques (P2.3 : noeuds, temps, mémoire) ───────────────
    nodes_expanded = 0   # noeuds extraits de OPEN (mesure de l'effort de recherche)
    peak_open = 1        # taille maximale atteinte par OPEN (empreinte mémoire)

    # ── Sélection de la fonction heuristique selon le mode ───────────────────
    h_func = heuristic_zero if use_h_zero else heuristic_manhattan

    # ─────────────────────────────────────────────────────────────────────────
    #  Boucle principale : tant qu'il reste des états à explorer
    # ─────────────────────────────────────────────────────────────────────────
    while open_set:

        # Suivre la taille maximale de OPEN pour mesurer l'empreinte mémoire
        peak_open = max(peak_open, len(open_set))

        # Extraire l'état avec la plus petite valeur f(n) de OPEN
        _, current = heapq.heappop(open_set)
        nodes_expanded += 1

        # ── Condition d'arrêt : état but atteint ─────────────────────────────
        if current == goal:
            # Reconstruction du chemin : remonter came_from de goal vers start
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()   # inverser pour obtenir l'ordre start → goal

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return path, g_score[goal], nodes_expanded, elapsed_ms, peak_open

        # Marquer current comme définitivement traité (ne sera plus inséré)
        closed.add(current)

        # ── Expansion : générer et évaluer tous les voisins accessibles ───────
        for nx, ny, cost in grid.neighbors(*current):
            neighbor = (nx, ny)

            # Calcul du coût tentatif pour atteindre neighbor via current
            tentative_g = g_score[current] + cost

            # Ignorer le voisin s'il est dans CLOSED avec un coût déjà optimal
            if neighbor in closed and tentative_g >= g_score.get(neighbor, float('inf')):
                continue

            # Si ce chemin est meilleur que tout chemin connu vers neighbor
            if tentative_g < g_score.get(neighbor, float('inf')):

                # Enregistrer le prédécesseur et le nouveau coût g
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # Calcul de la priorité f(n) selon le mode sélectionné
                if pure_greedy:
                    # Greedy Best-First : f = h(n)  — ignore le coût accumulé
                    f = h_func(neighbor, goal)
                else:
                    # A* standard ou Weighted : f = g(n) + w * h(n)
                    f = tentative_g + weight * h_func(neighbor, goal)

                # Insérer neighbor dans OPEN avec sa nouvelle priorité
                heapq.heappush(open_set, (f, neighbor))

    # ── Aucun chemin n'existe entre start et goal ─────────────────────────────
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    return None, float('inf'), nodes_expanded, elapsed_ms, peak_open