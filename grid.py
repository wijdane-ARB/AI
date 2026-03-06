"""
grid.py — Phase 1 : Définition de l'environnement (grille 2D)
================================================================
Modélise l'espace d'états sous forme d'une grille rectangulaire.

Chaque cellule libre est un état n ∈ S.
Les transitions définissent le graphe pondéré utilisé par A* et Markov.
Coût uniforme c(n, n') = 1 pour tous les déplacements valides.
"""


class Grid:
    """
    Grille 2D avec obstacles, état initial et état but.

    Attributs
    ---------
    width, height : dimensions de la grille
    obstacles     : ensemble des cellules bloquées (non franchissables)
    start         : état initial s0 = (x, y)
    goal          : état but g = (x, y)
    """

    def __init__(self, width, height, obstacles, start, goal):
        # Dimensions de la grille (nombre de colonnes et de lignes)
        self.width  = width
        self.height = height

        # Convertir la liste d'obstacles en ensemble pour lookup O(1)
        self.obstacles = set(obstacles)

        # État initial s0 et état but g (P1.1)
        self.start = start
        self.goal  = goal

    def is_free(self, x, y):
        """
        Vérifie si la cellule (x, y) est accessible.
        Une cellule est libre si :
          - elle est dans les bornes de la grille, ET
          - elle ne figure pas dans la liste des obstacles.
        """
        return (0 <= x < self.width
                and 0 <= y < self.height
                and (x, y) not in self.obstacles)

    def neighbors(self, x, y):
        """
        Retourne les voisins accessibles de (x, y) en 4-connexité.
        Directions : haut, bas, droite, gauche.
        Coût uniforme = 1 pour chaque déplacement (P1.2).

        Retourne : liste de (nx, ny, coût)
        """
        # Les 4 directions cardinales (4-voisins, grille uniforme)
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Filtrer pour ne garder que les cellules libres et dans les bornes
        return [(x + dx, y + dy, 1)
                for dx, dy in dirs
                if self.is_free(x + dx, y + dy)]

    def all_free_cells(self):
        """
        Retourne la liste de toutes les cellules libres de la grille.
        Utilisé pour construire l'espace d'états S de la chaîne de Markov.
        """
        return [(x, y)
                for x in range(self.width)
                for y in range(self.height)
                if self.is_free(x, y)]