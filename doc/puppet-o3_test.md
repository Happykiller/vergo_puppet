test_data = [
    # Paire 1
    [["air", "squat"], ["air", "squat"], 1.0],  # Vecteurs identiques
    # Paire 2
    [["push", "up"], ["push", "up"], 1.0],  # Vecteurs identiques
    # Paire 3
    [["man", "push", "up"], ["woman", "push", "up"], 0.85],  # Mots communs "push", "up"
    # Paire 4
    [["man", "squat", "dumbbell"], ["woman", "squat", "dumbbell"], 0.85],  # Mots communs "squat", "dumbbell"
    # Paire 5
    [["jump", "squat"], ["squat", "jump"], 0.9],  # Mêmes mots, ordre différent
    # Paire 6
    [["mountain", "climber"], ["mountain", "climber"], 1.0],  # Vecteurs identiques
    # Paire 7
    [["man", "dumbbell", "bicep", "curl"], ["woman", "dumbbell", "bicep", "curl"], 0.85],  # Mots communs "dumbbell", "bicep", "curl"
    # Paire 8
    [["push", "up"], ["pull", "up"], 0.5],  # Un mot commun "up"
    # Paire 9
    [["lunges", "step", "up"], ["step", "up", "knee", "lift"], 0.7],  # Mots communs "step", "up"
    # Paire 10
    [["burpee"], ["jump", "squat"], 0.3],  # Mots liés mais pas identiques
    # Paire 11
    [["plank", "shoulder", "tap"], ["plank", "shoulder", "tap"], 1.0],  # Vecteurs identiques
    # Paire 12
    [["man", "dumbbell", "deadlift"], ["woman", "kettlebell", "deadlift"], 0.7],  # Mot commun "deadlift"
    # Paire 13
    [["glute", "bridge"], ["hip", "thrust"], 0.4],  # Concepts similaires, mots différents
    # Paire 14
    [["high", "knees"], ["butt", "kick"], 0.2],  # Mouvements cardio, peu de mots communs
    # Paire 15
    [["lat", "pull", "down"], ["pull", "up"], 0.3],  # Un mot commun "pull"
    # Paire 16
    [["man", "tricep", "dip"], ["woman", "tricep", "kickback"], 0.5],  # Mot commun "tricep"
    # Paire 17
    [["jump", "rope"], ["high", "knees"], 0.2],  # Activités cardio, pas de mots communs
    # Paire 18
    [["box", "jump"], ["jump", "squat"], 0.6],  # Mot commun "jump"
    # Paire 19
    [["man", "bench", "press"], ["woman", "bench", "press"], 0.85],  # Mots communs "bench", "press"
    # Paire 20
    [["sitting", "hamstring", "stretch"], ["standing", "hamstring", "stretch"], 0.9],  # Mots communs "hamstring", "stretch"
    # Paire 21
    [["deadlift"], ["squat"], 0.3],  # Exercices différents, peu de mots communs
    # Paire 22
    [["plank"], ["plank", "exercise"], 0.9],  # Mot commun "plank", vecteur 2 a un mot supplémentaire
    # Paire 23
    [["yoga", "pose"], ["woman", "yoga", "stand", "stretch"], 0.4],  # Mot commun "yoga"
    # Paire 24
    [["core", "exercise"], ["plank", "shoulder", "tap"], 0.2],  # Concepts liés, peu de mots communs
    # Paire 25
    [["jump", "squat"], ["box", "jump"], 0.6],  # Mot commun "jump"
    # Paire 26
    [["kettlebell", "swing"], ["kettlebell", "swing"], 1.0],  # Vecteurs identiques
    # Paire 27
    [["lunge"], ["lunges", "step", "up"], 0.8],  # Mot commun "lunge" (singulier/pluriel)
    # Paire 28
    [["sit", "up"], ["crunch"], 0.3],  # Exercices similaires, mots différents
    # Paire 29
    [["pull", "up"], ["chin", "up"], 0.5],  # Mot commun "up"
    # Paire 30
    [["bicep", "curl"], ["hammer", "curl"], 0.5],  # Mot commun "curl"
]
