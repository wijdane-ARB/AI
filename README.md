# [cite_start]Planification Robuste sur Grille : A* et Chaînes de Markov [cite: 254]

[cite_start]Ce dépôt contient l'implémentation Python d'un mini-projet hybride combinant la recherche heuristique déterministe et la modélisation stochastique[cite: 259, 260, 261]. [cite_start]Il a été réalisé dans le cadre du Master Sciences de Données et Intelligence Artificielle (SDIA) [cite: 2] [cite_start]à l'ENSET Mohammedia [cite: 1][cite_start], sous l'encadrement de M. Mohamed MESTARI[cite: 6].

[cite_start]**Auteure :** Wijdane AARROUB [cite: 4]  
[cite_start]**Date :** 3 mars 2026 [cite: 5]

---

## 🎯 Objectif du Projet

[cite_start]Dans de nombreux systèmes autonomes, l'exécution d'une action n'est pas parfaitement déterministe (glissements, perturbations)[cite: 265]. [cite_start]L'objectif est double[cite: 267]:
1. [cite_start]**Planification :** Trouver un chemin optimal dans une grille avec obstacles à l'aide d'algorithmes de recherche heuristique (A*, UCS, Greedy)[cite: 260].
2. **Évaluation de la robustesse :** Modéliser l'incertitude d'exécution (déviations latérales) à l'aide de Chaînes de Markov à temps discret pour évaluer la probabilité réelle d'atteindre le but[cite: 261, 267].

## 📂 Structure du Code

Le projet est divisé en modules distincts[cite: 329]:
* [cite_start]`grid.py` : Définition de l'environnement (grille, obstacles, déplacements)[cite: 330].
* [cite_start]`astar.py` : Implémentation des algorithmes de recherche (A*, UCS, Greedy, Weighted A*)[cite: 330].
* [cite_start]`markov.py` : Construction de la matrice de transition stochastique, analyse des classes de communication (composantes fortement connexes), et calculs d'absorption exacts via la matrice fondamentale[cite: 331, 335].
* `utils.py` : Fonctions de visualisation (tracé des chemins, des pièges markoviens, et graphiques comparatifs).
* [cite_start]`experiments.py` : Script principal orchestrant les 5 phases du projet[cite: 329].
* [cite_start]`main.py` : Point d'entrée de l'application[cite: 329].

## 🛠️ Installation et Exécution

**Prérequis :** Python 3.x, `numpy`, `networkx`, `matplotlib`

```bash
# Cloner le dépôt
git clone [https://github.com/votre-nom-utilisateur/nom-du-repo.git](https://github.com/votre-nom-utilisateur/nom-du-repo.git)
cd nom-du-repo

# Installer les dépendances
pip install numpy networkx matplotlib

# Lancer les expérimentations
python main.py
```
## 📊 Résultats Expérimentaux (Grille Difficile)

Les expériences menées sur la grille difficile démontrent l'efficacité de l'approche :

* **Planification Déterministe :** L'algorithme A* avec l'heuristique de Manhattan trouve le chemin optimal (coût = 12) en développant 40 nœuds, illustrant le principe de dominance face à UCS (41 nœuds).
* **Expérience Weighted A* :** L'utilisation d'une pondération (w = 1.5) permet de réduire drastiquement l'espace d'exploration à seulement 13 nœuds (comportement équivalent à Greedy), tout en conservant ici l'optimalité.
* **Analyse Markovienne :** L'analyse de la matrice de transition (ε = 0.2) identifie exactement 14 classes de communication, dont une classe récurrente absorbante (le but) et une classe récurrente piège aux coordonnées (3,3), (3,4).
* **Validation par Simulation :** Probabilité d'absorption mathématique exacte de 97.42%. Taux de succès par simulation de Monte-Carlo (10 000 itérations) de 97.35%. Temps d'atteinte moyen de 15.0 pas.
## 📈 Visualisations
* **Comparaison des Algorithmes :** Graphique comparatif des coûts et nœuds développés pour A*, UCS, Greedy, et Weighted A*.
img src="comparison_algorithms.png" alt="Comparaison des Algorithmes" width="600"/>
* **Impact de ε :** Graphique montrant la probabilité d'atteindre le but en fonction de ε, illustrant la dégradation de la performance avec l'augmentation de l'incertitude.
img src="epsilon_impact.png" alt="Impact de ε" width="600"/>
* **Grille avec Chemin et Piège :** Visualisation de la grille difficile avec le chemin optimal trouvé par A* et les états pièges identifiés par l'analyse markovienne. 
img src="hard_grid_with_trap.png" alt="Grille Difficile avec Chemin et Piège" width="600"/>