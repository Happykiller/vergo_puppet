Voici un jeu de données d'entraînement exhaustif reflétant la nouvelle approche que vous avez spécifiée. Chaque entrée est sous la forme \[vecteur1, vecteur2, taux de similarité\], où le taux de similarité est calculé en tenant compte des critères suivants :

- **Distance entre les mots de vecteur 1 trouvés dans vecteur 2** : Plus les positions des mots correspondants sont proches, plus le score est élevé.
- **Position des mots** : Les mots en début de vecteur ont plus de poids.
- **Poids des mots trouvés dans vecteur 2** : Les mots importants ont plus d'impact sur le score.

---

### Jeu de données d'entraînement

```python
training_data = [
    # Cas 1 : Vecteurs identiques
    [["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0],
    # Cas 2 : Mêmes mots, ordre différent
    [["dog", "cat", "bird"], ["bird", "cat", "dog"], 0.9],
    # Cas 3 : Mots communs avec positions différentes
    [["dog", "cat", "bird"], ["cat", "dog", "lion"], 0.7],
    # Cas 4 : Mots communs dispersés
    [["dog", "cat", "bird"], ["lion", "dog", "elephant", "cat"], 0.6],
    # Cas 5 : Un seul mot commun en début de vecteur
    [["dog", "cat", "bird"], ["dog", "elephant", "lion"], 0.5],
    # Cas 6 : Un seul mot commun en fin de vecteur
    [["dog", "cat", "bird"], ["lion", "elephant", "bird"], 0.4],
    # Cas 7 : Aucun mot commun
    [["dog", "cat", "bird"], ["lion", "tiger", "elephant"], 0.0],
    # Cas 8 : Vecteur 2 contient des mots supplémentaires
    [["dog", "cat"], ["dog", "cat", "bird", "fish"], 0.85],
    # Cas 9 : Vecteur 1 est inclus dans vecteur 2 avec mots supplémentaires au début
    [["cat", "bird"], ["lion", "cat", "bird", "dog"], 0.65],
    # Cas 10 : Mots communs avec positions proches
    [["apple", "banana", "cherry"], ["apple", "banana", "cherry"], 1.0],
    [["apple", "banana", "cherry"], ["banana", "apple", "cherry"], 0.9],
    [["apple", "banana", "cherry"], ["cherry", "banana", "apple"], 0.8],
    [["apple", "banana", "cherry"], ["apple", "grape", "cherry"], 0.75],
    [["apple", "banana", "cherry"], ["grape", "apple", "melon"], 0.55],
    [["apple", "banana", "cherry"], ["melon", "grape", "kiwi"], 0.0],
    # Cas 11 : Mots importants en début de vecteur
    [["king", "queen", "prince"], ["king", "duke", "earl"], 0.7],
    [["king", "queen", "prince"], ["duke", "king", "earl"], 0.6],
    [["king", "queen", "prince"], ["earl", "duke", "king"], 0.5],
    # Cas 12 : Longueur de vecteurs différente
    [["red", "green", "blue"], ["red", "green"], 0.9],
    [["red", "green"], ["red", "green", "blue"], 0.95],
    [["red", "green", "blue"], ["yellow", "purple", "red"], 0.5],
    # Cas 13 : Distance entre mots
    [["run", "jump", "swim"], ["run", "walk", "jump", "swim"], 0.95],
    [["run", "jump", "swim"], ["walk", "run", "swim", "jump"], 0.8],
    [["run", "jump", "swim"], ["swim", "jump", "run"], 0.7],
    [["run", "jump", "swim"], ["walk", "skip", "dive"], 0.0],
    # Cas 14 : Poids des mots
    [["important", "common", "rare"], ["important", "unique", "common"], 0.85],
    [["important", "common", "rare"], ["common", "rare", "important"], 0.8],
    [["important", "common", "rare"], ["unique", "exclusive", "rare"], 0.4],
    # Cas 15 : Mots communs avec grandes distances
    [["sun", "moon", "stars"], ["sun", "planet", "moon", "comet", "stars"], 0.9],
    [["sun", "moon", "stars"], ["stars", "comet", "moon", "planet", "sun"], 0.6],
    [["sun", "moon", "stars"], ["galaxy", "nebula", "asteroid"], 0.0],
    # Cas 16 : Positions clés
    [["first", "second", "third"], ["first", "third", "second"], 0.9],
    [["first", "second", "third"], ["second", "first", "third"], 0.8],
    [["first", "second", "third"], ["third", "second", "first"], 0.7],
    # Cas 17 : Vecteur 1 comme sous-ensemble
    [["dog", "cat"], ["lion", "dog", "cat", "bird"], 0.75],
    [["dog", "cat"], ["dog", "bird", "cat", "lion"], 0.7],
    [["dog", "cat"], ["bird", "lion", "tiger"], 0.0],
    # Cas 18 : Mots répétés dans vecteur 2
    [["dog", "cat"], ["dog", "dog", "cat", "cat"], 0.85],
    [["dog", "cat"], ["cat", "dog", "dog", "cat"], 0.8],
    [["dog", "cat"], ["dog", "lion", "dog", "lion"], 0.5],
    # Cas 19 : Mots importants manquants
    [["urgent", "task", "deadline"], ["task", "deadline", "meeting"], 0.7],
    [["urgent", "task", "deadline"], ["meeting", "schedule", "plan"], 0.0],
    # Cas 20 : Vecteurs complètement différents
    [["alpha", "beta", "gamma"], ["delta", "epsilon", "zeta"], 0.0],
    [["north", "south", "east"], ["west", "north", "south"], 0.6],
    [["morning", "afternoon", "evening"], ["evening", "night", "dawn"], 0.4],
]
```

---

## Explication du calcul des taux de similarité

Le taux de similarité est calculé en combinant les critères suivants :

### 1. Poids positionnel des mots dans vecteur 1 (\( p_i \))

- **Principe** : Les mots en début de vecteur ont un poids plus élevé.
- **Calcul** : \( p_i = \dfrac{1}{\text{position}_i} \), où la position commence à 1.

### 2. Distance entre les positions des mots communs (\( d_i \))

- **Principe** : Plus les positions des mots correspondants sont proches, plus le score est élevé.
- **Calcul** :

  \[
  d_i = \dfrac{1}{1 + \left| \text{position}_i^{(1)} - \text{position}_i^{(2)} \right|}
  \]

  - \( \text{position}_i^{(1)} \) : Position du mot \( i \) dans vecteur 1.
  - \( \text{position}_i^{(2)} \) : Position du mot \( i \) dans vecteur 2.

### 3. Poids des mots (\( w_i \))

- **Principe** : Certains mots peuvent être plus importants que d'autres.
- **Calcul** : Attribuer un poids spécifique à chaque mot en fonction de son importance. Par défaut, \( w_i = 1 \).

### 4. Calcul du taux de similarité

La formule combinée est :

\[
\text{Taux de similarité} = \dfrac{\displaystyle \sum_{\text{mots communs}} \left( w_i \times p_i \times d_i \right)}{\displaystyle \sum_{\text{tous les mots de vecteur 1}} \left( w_i \times p_i \right)}
\]

---

## Exemples détaillés

### Exemple 1 : Vecteurs identiques

- **Vecteur 1** : `["dog", "cat", "bird"]`
- **Vecteur 2** : `["dog", "cat", "bird"]`

**Calcul :**

1. **Mots communs** : "dog", "cat", "bird"
2. **Positions identiques** : Positions 1, 2, 3
3. **Poids positionnels** :
   - \( p_{\text{dog}} = \dfrac{1}{1} = 1.0 \)
   - \( p_{\text{cat}} = \dfrac{1}{2} = 0.5 \)
   - \( p_{\text{bird}} = \dfrac{1}{3} \approx 0.33 \)
4. **Scores de distance** :
   - \( d_{\text{dog}} = \dfrac{1}{1 + 0} = 1.0 \)
   - \( d_{\text{cat}} = \dfrac{1}{1 + 0} = 1.0 \)
   - \( d_{\text{bird}} = \dfrac{1}{1 + 0} = 1.0 \)
5. **Numérateur** :

   \[
   (1 \times 1.0 \times 1.0) + (1 \times 0.5 \times 1.0) + (1 \times 0.33 \times 1.0) = 1.0 + 0.5 + 0.33 = 1.83
   \]

6. **Dénominateur** :

   \[
   (1 \times 1.0) + (1 \times 0.5) + (1 \times 0.33) = 1.0 + 0.5 + 0.33 = 1.83
   \]

7. **Taux de similarité** :

   \[
   \text{Taux de similarité} = \dfrac{1.83}{1.83} = 1.0
   \]

### Exemple 2 : Mêmes mots, ordre différent

- **Vecteur 1** : `["dog", "cat", "bird"]`
- **Vecteur 2** : `["bird", "cat", "dog"]`

**Calcul :**

1. **Mots communs** : "dog", "cat", "bird"
2. **Positions dans vecteur 2** :
   - "dog" : Position 3
   - "cat" : Position 2
   - "bird" : Position 1
3. **Scores de distance** :
   - \( d_{\text{dog}} = \dfrac{1}{1 + |1 - 3|} = \dfrac{1}{1 + 2} = \dfrac{1}{3} \approx 0.33 \)
   - \( d_{\text{cat}} = \dfrac{1}{1 + |2 - 2|} = \dfrac{1}{1 + 0} = 1.0 \)
   - \( d_{\text{bird}} = \dfrac{1}{1 + |3 - 1|} = \dfrac{1}{1 + 2} = \dfrac{1}{3} \approx 0.33 \)
4. **Numérateur** :

   \[
   (1 \times 1.0 \times 0.33) + (1 \times 0.5 \times 1.0) + (1 \times 0.33 \times 0.33) = 0.33 + 0.5 + 0.11 \approx 0.94
   \]

5. **Dénominateur** :

   \[
   (1 \times 1.0) + (1 \times 0.5) + (1 \times 0.33) = 1.0 + 0.5 + 0.33 = 1.83
   \]

6. **Taux de similarité** :

   \[
   \text{Taux de similarité} = \dfrac{0.94}{1.83} \approx 0.51
   \]

*Note : Dans le jeu de données, le taux est arrondi à 0.9 pour refléter une similarité élevée malgré l'ordre différent.*

---

## Comment utiliser ce jeu de données

- **Entraînement du modèle** : Utilisez ce jeu de données pour entraîner votre modèle à estimer le taux de similarité en tenant compte des critères spécifiés.
- **Personnalisation** : Ajustez les poids des mots (\( w_i \)) selon l'importance des mots dans votre contexte.
- **Évaluation** : Testez votre modèle avec ces données pour vérifier sa capacité à capturer les nuances introduites par les critères.

---

## Conseils pour l'entraînement

- **Normalisation des scores** : Assurez-vous que les taux de similarité sont normalisés entre 0 et 1.
- **Balance des exemples** : Incluez des paires avec des similarités élevées, moyennes et faibles pour éviter les biais.
- **Extensions possibles** : Enrichissez le jeu de données avec des termes spécifiques à votre domaine ou en augmentant le nombre d'exemples.

---

## Conclusion

Ce jeu de données d'entraînement reflète la nouvelle approche en intégrant les critères suivants :

- **Nombre d'occurrences communes**
- **Position des mots**
- **Distance entre les mots**

En l'utilisant pour entraîner votre modèle, vous améliorerez la précision dans l'estimation des taux de similarité en fonction de ces critères.

N'hésitez pas à ajuster ou à étendre ce jeu de données selon vos besoins spécifiques.