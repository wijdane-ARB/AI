# Planification Robuste sur Grille : A* + Chaînes de Markov

**Auteure :** Wijdane AARROUB  
**Date :** 3 mars 2026  
**Filière :** Master Sciences de Données et Intelligence Artificielle (SDIA) – ENSET Mohammedia  
**Encadrant :** M. Mohamed MESTARI

## Objectif
Implémentation hybride : planification optimale avec **A*** (heuristique Manhattan admissible) + modélisation de l’incertitude avec **Chaînes de Markov** (glissement ε).

## Structure du projet
- `grid.py` – Environnement
- `astar.py` – A*, UCS, Greedy, h=0
- `markov.py` – Matrice P, classes de communication, probabilité d’absorption **exacte**
- `utils.py` – Graphiques + visualisation grille + piège
- `experiments.py` – Phases 1 à 5 + E.1/E.2/E.3
- `main.py` – Point d’entrée

## Résultats (extrait console)
- Grille Difficile : Probabilité exacte = **0.9742** (simulation ≈ 0.9735)
- Classe piège détectée : `[(3,3), (3,4)]` → **RÉCURRENT (PIÈGE)**

## Visualisations générées
![Comparaison](comparison_algorithms.png)  
![Impact ε](epsilon_impact.png)  
![Grille + Piège](hard_grid_with_trap.png)

## Exécution
```bash
pip install numpy networkx matplotlib
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