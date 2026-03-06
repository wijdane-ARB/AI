"""
markov.py — Phases 3, 4, 5 : Chaînes de Markov et analyse probabiliste
========================================================================
Construit la matrice de transition P à partir du chemin A* (politique),
calcule π^(n) = π^(0) · P^n, identifie les classes de communication,
calcule les probabilités d'absorption exactes (matrice fondamentale)
et simule des trajectoires Monte-Carlo.

Modèle d'incertitude (paramètre ε) :
  - Avec probabilité (1 - ε) : l'agent suit l'action prescrite.
  - Avec probabilité ε/2     : déviation vers chaque côté latéral.
  - En cas de collision      : rebond (auto-boucle sur l'état courant).
"""

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Construction de la matrice de transition P
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(grid, path, epsilon=0.1):
    """
    Construit la matrice stochastique P de la chaîne de Markov (P3.1, P3.2).

    Politique induite par le chemin A* :
      - Sur le chemin A* → suivre le prochain état du chemin.
      - Hors chemin      → voisin libre minimisant h(s, goal) (Manhattan).

    Modèle de glissement ε (P1.3) :
      - p(s, s') = 1 - ε   pour l'action principale (politique).
      - p(s, s'') = ε/2    pour chaque déviation latérale.
      - Auto-boucle si la direction cible est un obstacle ou hors bornes.

    États absorbants (section 3.3 du rapport) :
      - GOAL : p(goal, goal) = 1  (absorbe définitivement).
      - FAIL  : états piège sans chemin vers GOAL (classe récurrente non-GOAL).

    Paramètres
    ----------
    grid    : objet Grid
    path    : chemin A* optimal (liste d'états)
    epsilon : taux de glissement ε ∈ [0, 1]  (P1.3)

    Retourne
    --------
    (P, states, state_to_idx)
      P             : matrice numpy (n×n), stochastique par lignes
      states        : liste ordonnée des états
      state_to_idx  : dictionnaire état → indice de ligne/colonne
    """

    # ── Construire la politique à partir du chemin A* (P3.1) ─────────────────
    # policy[s] = s' signifie "depuis s, l'action recommandée mène à s'"
    policy = {}
    for i in range(len(path) - 1):
        policy[path[i]] = path[i + 1]

    # ── Définir l'espace d'états S = toutes les cellules libres ──────────────
    states = grid.all_free_cells()

    # S'assurer que GOAL est bien dans la liste des états
    if grid.goal not in states:
        states.append(grid.goal)

    # Créer un index bidirectionnel état <-> indice pour construire P
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    # Initialiser la matrice P à zéro (n x n)
    P = np.zeros((n, n))

    # Heuristique locale pour la politique hors-chemin
    def local_heuristic(a, b):
        """Distance de Manhattan pour orienter l'agent hors du chemin A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ── Remplissage de P : une ligne par état ─────────────────────────────────
    for s in states:
        i = state_to_idx[s]

        # GOAL est un état absorbant : l'agent reste dans GOAL pour toujours
        if s == grid.goal:
            P[i, i] = 1.0   # p(goal, goal) = 1
            continue

        # Déterminer l'action prescrite par la politique pour cet état
        if s in policy:
            # L'état est sur le chemin A* : suivre le chemin
            next_s = policy[s]
        else:
            # L'état est hors chemin : choisir le voisin le plus proche du but
            neighbors = [(cx, cy) for cx, cy, _ in grid.neighbors(*s)]
            if neighbors:
                next_s = min(neighbors, key=lambda p: local_heuristic(p, grid.goal))
            else:
                # Aucun voisin libre → l'agent reste sur place (piège)
                next_s = s

        # Calculer la direction principale (action prescrite)
        dx, dy = next_s[0] - s[0], next_s[1] - s[1]

        # Définir les directions : [principale, latérale_1, latérale_2]
        # La déviation latérale est perpendiculaire à la direction principale
        if   (dx, dy) == (1,  0): dirs = [(1, 0),  (0, 1),  (0, -1)]  # droite
        elif (dx, dy) == (-1, 0): dirs = [(-1, 0), (0, 1),  (0, -1)]  # gauche
        elif (dx, dy) == (0,  1): dirs = [(0, 1),  (1, 0),  (-1, 0)]  # haut
        elif (dx, dy) == (0, -1): dirs = [(0, -1), (1, 0),  (-1, 0)]  # bas
        else:                     dirs = [(0, 0)]                      # immobile

        # Probabilités associées à chaque direction
        # [1-ε, ε/2, ε/2] si 3 directions, [1.0] si immobile
        probs = ([1 - epsilon] + [epsilon / 2] * (len(dirs) - 1)
                 if len(dirs) > 1 else [1.0])
        probs = probs[:len(dirs)]   # sécurité : tronquer si nécessaire

        # Distribuer les probabilités dans la matrice P
        for d, p in zip(dirs, probs):
            tx, ty = s[0] + d[0], s[1] + d[1]
            target = (tx, ty)

            if grid.is_free(tx, ty) or target == grid.goal:
                # Destination accessible : affecter la probabilité à la colonne cible
                j = state_to_idx.get(target, i)
                P[i, j] += p
            else:
                # Destination bloquée (obstacle ou hors bornes) : rebond sur place
                P[i, i] += p

    # ── Vérification : P doit être stochastique (somme des lignes = 1) ────────
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8), "Matrice P non stochastique !"

    return P, states, state_to_idx


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — Évolution de la distribution π^(n) = π^(0) · P^n
# ─────────────────────────────────────────────────────────────────────────────

def compute_pi_n(P, start_idx, n_steps=30):
    """
    Calcule π^(n) par itération de Chapman–Kolmogorov (P3.3).

    Formule : π^(n) = π^(0) · P^n  (appliqué récursivement)
    À chaque pas : π^(n) = π^(n-1) · P

    π^(n)[goal] donne la probabilité d'être dans GOAL après n pas,
    en partant de s0 avec certitude (π^(0)[s0] = 1).

    Paramètres
    ----------
    P        : matrice de transition (n×n)
    start_idx: indice de l'état initial s0
    n_steps  : nombre de pas de temps à calculer

    Retourne
    --------
    Liste de n_steps vecteurs π^(n) (numpy arrays)
    """
    n = P.shape[0]

    # Initialisation : π^(0) — masse de probabilité concentrée sur s0
    pi = np.zeros(n)
    pi[start_idx] = 1.0   # P(X_0 = s0) = 1

    results = []
    for step in range(1, n_steps + 1):
        # Mise à jour Chapman-Kolmogorov : π^(n) = π^(n-1) · P
        pi = pi @ P
        results.append(pi.copy())   # sauvegarder π^(n) pour analyse

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5 — Simulation Monte-Carlo
# ─────────────────────────────────────────────────────────────────────────────

def simulate_markov(P, start_idx, goal_idx, n_sim=10000, max_steps=300):
    """
    Simulation Monte-Carlo de N trajectoires de la chaîne de Markov (P5.1, P5.2).

    Chaque trajectoire part de s0 et s'arrête dès que :
      - L'agent atteint GOAL  → succès.
      - L'agent tombe dans un piège (état non-GOAL sans chemin vers GOAL) → échec.
      - max_steps est atteint sans absorption.

    Paramètres
    ----------
    P         : matrice de transition (n×n)
    start_idx : indice de s0
    goal_idx  : indice de GOAL
    n_sim     : nombre de trajectoires simulées
    max_steps : limite de pas par trajectoire

    Retourne
    --------
    (p_success, avg_time, times_success, times_fail)
      p_success     : P̂(GOAL) empirique
      avg_time      : temps moyen d'atteinte de GOAL
      times_success : liste des temps d'absorption en GOAL
      times_fail    : liste des temps d'absorption dans les pièges
    """

    successes  = 0
    times_suc  = []   # temps d'atteinte de GOAL pour chaque trajectoire réussie
    times_fail = []   # temps avant capture par un piège
    n_states   = len(P)

    # ── Pré-calcul des états pièges (FAIL) ───────────────────────────────────
    # Un état est un piège s'il n'existe aucun chemin vers GOAL dans le graphe P
    trap_states = set()
    for i in range(n_states):
        if i == goal_idx:
            continue   # GOAL n'est pas un piège

        # Recherche en profondeur pour vérifier l'accessibilité de GOAL
        reachable = False
        visited   = set()
        stack     = [i]
        while stack:
            cur = stack.pop()
            if cur == goal_idx:
                reachable = True   # GOAL atteignable depuis i
                break
            if cur in visited:
                continue
            visited.add(cur)
            # Explorer tous les successeurs possibles de cur
            for j in range(n_states):
                if P[cur, j] > 1e-8 and j not in visited:
                    stack.append(j)

        if not reachable:
            trap_states.add(i)   # i est un état piège (FAIL)

    # ── Boucle de simulation : N trajectoires indépendantes ──────────────────
    for _ in range(n_sim):
        state = start_idx   # Partir de s0 au début de chaque trajectoire

        for t in range(1, max_steps + 1):
            # Tirer le prochain état selon la distribution P[state, :]
            state = np.random.choice(n_states, p=P[state])

            if state == goal_idx:
                # Trajectoire réussie : enregistrer le temps d'atteinte
                successes += 1
                times_suc.append(t)
                break

            if state in trap_states:
                # Trajectoire échouée : absorbé par un piège
                times_fail.append(t)
                break

    # ── Calcul des statistiques empiriques (P5.2) ─────────────────────────────
    p_success = successes / n_sim
    avg_time  = float(np.mean(times_suc)) if times_suc else float('inf')

    return p_success, avg_time, times_suc, times_fail


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4 — Calcul exact des probabilités d'absorption (matrice fondamentale)
# ─────────────────────────────────────────────────────────────────────────────

def compute_exact_absorption(P, start_idx, goal_idx, states, state_to_idx):
    """
    Calcule P(GOAL) et P(FAIL) exactement par décomposition canonique (P4.3).

    Méthode de la matrice fondamentale :
      Décomposition : P = [I  0]   (classes récurrentes)
                          [R  Q]   (états transitoires)

      Matrice fondamentale : N = (I - Q)^{-1}
        N[i, j] = nombre moyen de fois que la chaîne visite j avant absorption,
                  en partant de l'état transitoire i.

      Matrice d'absorption : B = N · R
        B[i, k] = probabilité d'être absorbé par la classe k
                  en partant de l'état transitoire i.

    Paramètres
    ----------
    P           : matrice de transition (n×n)
    start_idx   : indice de s0
    goal_idx    : indice de GOAL
    states      : liste des états
    state_to_idx: dictionnaire état → indice

    Retourne
    --------
    (p_goal, p_fail) : probabilités d'absorption en GOAL et dans les pièges
    """

    n = P.shape[0]

    # ── Construction du graphe dirigé des transitions ─────────────────────────
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-8:
                G.add_edge(i, j)   # arc i → j si p(i,j) > 0

    # ── Identification des composantes fortement connexes (CFC) ──────────────
    # Une CFC est récurrente si aucun arc ne la quitte
    scc = list(nx.strongly_connected_components(G))

    recurrent_classes = []   # classes récurrentes (absorbantes)
    goal_class = None         # classe contenant GOAL

    for component in scc:
        c_list = list(component)

        # Vérifier si la composante a des arcs sortants vers d'autres CFC
        has_outgoing = any(
            P[i, j] > 1e-8
            for i in c_list
            for j in range(n)
            if j not in c_list
        )

        if not has_outgoing:
            # Pas d'arc sortant → classe récurrente (absorbante)
            recurrent_classes.append(c_list)
            if goal_idx in c_list:
                goal_class = c_list   # mémoriser la classe de GOAL

    # Cas dégénéré : GOAL n'est dans aucune classe absorbante
    if not goal_class:
        return 0.0, 1.0

    # ── Séparation états transitoires / récurrents ────────────────────────────
    all_recurrent = set(s for cls in recurrent_classes for s in cls)
    transient     = [i for i in range(n) if i not in all_recurrent]

    # Si start est déjà dans une classe récurrente, la réponse est triviale
    if start_idx in all_recurrent:
        p_goal = 1.0 if start_idx in goal_class else 0.0
        return p_goal, 1.0 - p_goal

    # ── Construction des matrices Q et R ──────────────────────────────────────
    num_trans = len(transient)    # nombre d'états transitoires
    num_abs   = len(recurrent_classes)   # nombre de classes absorbantes

    # Q[i, j] : probabilité de passer de l'état transitoire i à j
    Q = np.zeros((num_trans, num_trans))

    # R[i, k] : probabilité de passer de l'état transitoire i à la classe k
    R = np.zeros((num_trans, num_abs))

    # Mapping : ancien indice global → nouvel indice dans la sous-matrice
    trans_map = {old: new for new, old in enumerate(transient)}

    for i_idx, i in enumerate(transient):
        for j in range(n):
            prob = P[i, j]
            if prob < 1e-10:
                continue   # ignorer les transitions négligeables

            if j in trans_map:
                # Transition d'un état transitoire à un autre → remplir Q
                Q[i_idx, trans_map[j]] += prob
            else:
                # Transition vers un état récurrent → remplir R
                abs_id = next(k for k, cls in enumerate(recurrent_classes) if j in cls)
                R[i_idx, abs_id] += prob

    # ── Calcul de la matrice fondamentale N = (I - Q)^{-1} ───────────────────
    try:
        N = np.linalg.inv(np.eye(num_trans) - Q)

        # Matrice d'absorption B = N · R
        B = N @ R

        # Extraire P(GOAL) et P(FAIL) pour l'état initial s0
        s_idx        = trans_map[start_idx]
        goal_class_id = recurrent_classes.index(goal_class)

        p_goal = B[s_idx, goal_class_id]   # probabilité d'être absorbé par GOAL
        p_fail = 1.0 - p_goal               # probabilité d'être absorbé par un piège

        return p_goal, p_fail

    except np.linalg.LinAlgError:
        # La matrice (I - Q) est singulière (cas rare)
        print("Erreur : matrice singulière lors de l'inversion de (I - Q).")
        return 0.0, 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4 — Analyse des classes de communication
# ─────────────────────────────────────────────────────────────────────────────

def analyze_markov_classes(P, states, state_to_idx, goal_idx):
    """
    Identifie et affiche les classes de communication de la chaîne (P4.1, P4.2).

    Classification :
      - RÉCURRENT ABSORBANT – GOAL : classe contenant l'état but.
      - RÉCURRENT ABSORBANT – FAIL : autre classe sans arc sortant (piège).
      - TRANSITOIRE                : classe avec au moins un arc sortant.
    """

    # ── Construire le graphe orienté des transitions ──────────────────────────
    G = nx.DiGraph()
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-8:
                # Utiliser les états réels (pas les indices) comme noeuds
                G.add_edge(states[i], states[j])

    # ── Calculer les composantes fortement connexes (Kosaraju/Tarjan) ─────────
    scc = list(nx.strongly_connected_components(G))

    print("=== Phase 4 — Classes de communication ===")
    print(f"Nombre de classes : {len(scc)}")

    for component in scc:
        c = list(component)

        # Vérifier si la composante a des arcs sortants (= transitoire)
        outgoing = any(
            P[state_to_idx[s], state_to_idx[t]] > 1e-8
            for s in c for t in states if t not in c
        )

        # Étiqueter la classe selon sa nature
        if any(state_to_idx[s] == goal_idx for s in c):
            status = "RÉCURRENT ABSORBANT — GOAL"
        elif not outgoing:
            # Classe fermée mais sans GOAL → piège (état FAIL)
            status = "RÉCURRENT ABSORBANT — FAIL (piège)"
        else:
            # Des transitions sortent de cette classe → états transitoires
            status = "TRANSITOIRE"

        print(f"  {status} : {c}")