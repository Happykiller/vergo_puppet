import random
import json
from typing import List, Tuple
import datetime

def get_timestamp() -> str:
    """
    Retourne la date/heure courante au format YYMMDDHHMI.
    Exemple : 2309271545 pour 2023-09-27 15:45.
    """
    now = datetime.datetime.now()
    return now.strftime("%y%m%d%H%M")

########################################################
# 1) TA LISTE INTÉGRALE DE PAIRES (search, best_match)
########################################################
ENTIRE_DATASET = [
[[ "man", "jump", "jack", "star" ], [ "man", "jump", "jack", "star" ]],
[[ "woman", "jump","jack","star"], ["woman","jump","jack"]],
[["man","arm","circle"], ["man","arm","circle"]],
[["woman","arm","circle"], ["woman","arm","circle"]],
[["man","bodyweight","squat"], ["man","bodyweight","squat"]],
[["woman","bodyweight","squat"], ["woman","bodyweight","squat"]],
[["man","high","knee"], ["man","high","knee","wall","against"]],
[["woman","high","knee"], ["woman","high","knee","forward","lift","jog","stance"]],
[["man","dynamic","stretch"], ["man","dynamic","hamstring","stretch"]],
[["woman","dynamic","stretch"], ["woman","standing","open","gate","dynamic","stretch","opener"]],
[["man","burpee"], ["man","chest","floor","burpee"]],
[["woman","burpee"], ["woman","burpee"]],
[["man","mountain","climber"], ["man","mountain","climber"]],
[["woman","mountain","climber"], ["woman","mountain","climber"]],
[["man","jump","squat"], ["man","jump","squat"]],
[["woman","jump","squat"], ["woman","jump","squat"]],
[["man","push","up","rotation"], ["man","push","up"]],
[["woman","push","up","rotation"], ["push","up","rotation"]],
[["man","twist","abdominal"], ["man","twist","abdominal","excercise"]],
[["woman","twist","abdominal"], ["woman","supine","spinal","twist","stand","ii","supta","abdominal","jathara","parivartanasana"]],
[["man","jump","lunge"], ["man","alternating","lunge","jump"]],
[["woman","jump","lunge"], ["woman","explosive","jump","alternating","lunge","step","four","lower","body","hamstring"]],
[["man","shoulder","stretch"], ["man","reverse","shoulder","stretch"]],
[["woman","shoulder","stretch"], ["woman","shoulder","stretch","long","resistance","strap"]],
[["man","quad","stretch"], ["man","standing","quad","stretch","yoga","wall"]],
[["woman","quad","stretch"], ["woman","standing","quad","stretch"]],
[["man","hamstring","stretch"], ["man","hamstring","stretch","elastic","strap"]],
[["woman","hamstring","stretch"], ["woman","hamstring","stretch"]],
[["man","child","stand"], ["man","child","stand","stretch"]],
[["woman","child","stand"], ["woman","child","stand","stretch"]],
[["man","jump","squat"], ["man","jump","squat"]],
[["woman","jump","squat"], ["woman","jump","squat"]],
[["man","foot","swing"], ["man","lat","foot","hip","swing"]],
[["woman","foot","swing"], ["woman","lat","foot","swing"]],
[["man","butt","kick"], ["man","butt","kick","character","setu"]],
[["woman","butt","kick"], ["butt","kick"]],
[["man","air","squat"], ["man","air","squat","ii","step","forward"]],
[["woman","air","squat"], ["air","squat"]],
[["man","glute","bridge"], ["man","bench","glute","bridge","bar"]],
[["woman","glute","bridge"], ["woman","chest","flye","glute","bridge"]],
[["man","bench","tricep","dip"], ["man","bench","tricep","dip"]],
[["woman","bench","tricep","dip"], ["woman","bench","tricep","dip"]],
[["man","tricep","extension","dumbbell"], ["man","dumbbell","overhead","tricep","extension"]],
[["woman","tricep","extension","dumbbell"], ["woman","dumbbell","tricep","extension"]],
[["man","diamond","push","up"], ["man","diamond","push","up","four","tricep","chest"]],
[["woman","diamond","push","up"], ["woman","diamond","pyramid","push","up"]],
[["man","tricep","kickback","dumbbell"], ["man","tricep","kickback","dumbbell"]],
[["woman","tricep","kickback","dumbbell"], ["tricep","kickback","dumbbell"]],
[["man","squat","dumbbell"], ["man","squat","dumbbell"]],
[["woman","squat","dumbbell"], ["woman","squat","dumbbell"]],
[["man","stepping","lunge","dumbbell"], ["man","dumbbell","stepping","lunge"]],
[["woman","stepping","lunge","dumbbell"], ["woman","body","dumbbell","stepping","lunge"]],
[["man","step","up","plyo","boxing","dumbbell"], ["step","up","plyo","boxing","dumbbell"]],
[["woman","step","up","plyo","boxing","dumbbell"], ["step","up","plyo","boxing","dumbbell"]],
[["man","glute","stretch"], ["man","pigeon","glute","stretch"]],
[["woman","glute","stretch"], ["woman","gluteus","glute","stretch"]],
[["man","bosu","ball","bridge","hip","lift","glute"], ["man","bosu","ball","bridge","hip","lift","glute"]],
[["woman","bosu","ball","bridge","hip","lift","glute"], ["woman","bosu","ball","bridge","hip","lift","glute"]],
[["man","supine","spinal","twist"], ["man","supta","matsyendrasana","supine","spinal","twist","stand"]],
[["woman","supine","spinal","twist"], ["woman","supine","spinal","twist","stand","ii","supta","abdominal","jathara","parivartanasana"]],
[["man","warm","up","hip","circle"], ["man","hip","lift","butt","bridge","curling","shoulder","reach","bodyweight"]],
[["woman","warm","up","hip","circle"], ["woman","hip","circle"]],
[["man","jump","rope"], ["man","jump","rope","cardio"]],
[["woman","jump","rope"], ["woman","jump","rope","cardio"]],
[["man","overhead","dumbbell","shoulder","press"], ["man","overhead","dumbbell","shoulder","press"]],
[["woman","overhead","dumbbell","shoulder","press"], ["woman","dumbbell","overhead","shoulder","press"]],
[["man","lat","shoulder","dumbbell","lift","strength","partial"], ["man","lat","shoulder","dumbbell","lift","strength","partial"]],
[["woman","lat","shoulder","dumbbell","lift","strength","partial"], ["woman","lat","shoulder","dumbbell","lift","strength","partial"]],
[["man","folded","up","reverse","flye"], ["man","dumbbell","folded","up","reverse","flye"]],
[["woman","folded","up","reverse","flye"], ["woman","dumbbell","folded","up","chest","salamba","reverse","flye"]],
[["man","standing","dumbbell","bicep","curl"], ["man","standing","dumbbell","bicep","curl"]],
[["woman","standing","dumbbell","bicep","curl"], ["woman","dumbbell","bicep","curl","different","layer","character"]],
[["man","tricep","dip"], ["man","bench","tricep","dip"]],
[["woman","tricep","dip"], ["woman","tricep","dip"]],
[["man","dumbbell","bicep","hammer","curl"], ["man","dumbbell","bicep","hammer","curl","different","layer","character"]],
[["woman","dumbbell","bicep","hammer","curl"], ["woman","dumbbell","bicep","hammer","curl","different","layer","character"]],
[["man","dumbbell","bench","press","chest"], ["man","dumbbell","bench","press","chest"]],
[["woman","dumbbell","bench","press","chest"], ["woman","dumbbell","chest","press"]],
[["man","dumbbell","flye"], ["man","bench","dumbbell","flye","overhead"]],
[["woman","dumbbell","flye"], ["woman","dumbbell","folded","up","chest","salamba","reverse","flye"]],
[["man","push","up"], ["man","push","up"]],
[["woman","push","up"], ["woman","asymmetrical","push","up"]],
[["man","dynamic","chest","stretch"], ["man","chest","stretch","wall","standing","one","arm"]],
[["woman","dynamic","chest","stretch"], ["woman","chest","stretch"]],
[["man","shrug","stretch"], ["man","stretch","up"]],
[["woman","shrug","stretch"], ["woman","stretch","up","machine","assisted"]],
[["man","chest","stretch"], ["man","chest","stretch","wall","standing","one","arm"]],
[["woman","chest","stretch"], ["woman","chest","stretch"]],
[["man","tricep","stretch"], ["man","overhead","tricep","stretch"]],
[["woman","tricep","stretch"], ["woman","tricep","stretch"]],
[["man","bicep","stretch"], ["man","stretch","up"]],
[["woman","bicep","stretch"], ["woman","bicep","stretch"]],
[["man","bench","jump","up"], ["man","bench","hop","boxing","jump","up","sport"]],
[["woman","bench","jump","up"], ["woman","sitting","bench","foot","stretch","knee","up"]],
[["man","warm","up","stepping"], ["man","plank","push","up","movement","stepping","down","ab"]],
[["woman","warm","up","stepping"], ["woman","bosu","ball","plank","push","up","stepping","down"]],
[["man","warm","up","hip","circle"], ["man","hip","lift","butt","bridge","curling","shoulder","reach","bodyweight"]],
[["woman","warm","up","hip","circle"], ["woman","hip","circle"]],
[["man","plank","row","weight"], ["man","plank","dumbbell","row","minimalistic","gym"]],
[["woman","plank","row","weight"], ["woman","plank","row","renegade"]],
[["man","lat","plank"], ["man","lat","plank","abdominal"]],
[["woman","lat","plank"], ["woman","lat","plank"]],
[["man","deadlift","weight"], ["man","dumbbell","deadlift"]],
[["woman","deadlift","weight"], ["woman","dumbbell","deadlift"]],
[["man","superman"], ["man","sitting"]],
[["woman","superman"], ["woman","superman","twist"]],
[["man","quadricep","stretch"], ["man","quadricep","stretch","cool","down","swing","stand","flexibility","improvement"]],
[["woman","quadricep","stretch"], ["woman","quadricep","stretch","cool","down","swing","stand","flexibility","improvement"]],
[["man","ischio","stretch"], ["man","stretch","up"]],
[["woman","ischio","stretch"], ["woman","stretch","up","machine","assisted"]],
[["man","shrug","stretch"], ["man","stretch","up"]],
[["woman","shrug","stretch"], ["woman","stretch","up","machine","assisted"]],
[["man","spinal","twist"], ["man","standing","spinal","twist","stand","katichakrasana"]],
[["woman","spinal","twist"], ["woman","chair","spinal","twist","ardha","matsyendrasana"]],
[["man","crunch"], ["man","crunch","abdominal"]],
[["woman","crunch"], ["woman","crunch"]],
[["man","prone","foot","lift"], ["man","prone","foot","lift","abdominal","body","dumbbell"]],
[["woman","prone","foot","lift"], ["woman","prone","foot","lift"]],
[["man","reverse","crunch"], ["man","incline","bench","reverse","crunch"]],
[["woman","reverse","crunch"], ["woman","crunch"]],
[["man","foot","lift"], ["man","hanging","foot","lift","lat","abdominal"]],
[["woman","foot","lift"], ["woman","foot","lift","reach","clap"]],
[["man","plank","foot","lift","ii","step"], ["tutorial","man","plank","foot","lift","ii","step"]],
[["woman","plank","foot","lift","ii","step"], ["tutorial","woman","plank","foot","lift","ii","step"]],
[["man","bicycle","crunch"], ["man","crunch","abdominal"]],
[["woman","bicycle","crunch"], ["woman","crunch"]],
[["man","bhujang","stretch"], ["man","bhujang","ab","stretch","old","ashwa","abdominal"]],
[["woman","bhujang","stretch"], ["woman","bhujang","ab","stretch","old","ashwa","abdominal","editable","file","layer"]],
[["man","sitting","oblique","stretch"], ["man","oblique","stretch"]],
[["woman","sitting","oblique","stretch"], ["woman","oblique","stretch"]],
[["man","sitting","overhead","dumbbell","tricep","extension"], ["man","sitting","overhead","dumbbell","tricep","extension"]],
[["woman","man","sitting","overhead","dumbbell","tricep","extension"], ["man","sitting","overhead","dumbbell","tricep","extension"]],
[["man","stretching","hip","circle"], ["man","lat","bend","stretching","hand","hip","sport"]],
[["woman","stretching","hip","circle"], ["woman","hip","circle"]],
[["man","forearm","plank"], ["man","plank","abdominal"]],
[["woman","forearm","plank"], ["woman","forearm","plank"]],
[["man","dead","bug"], ["man","dead","bug","abdominal","editable","file","layer"]],
[["woman","dead","bug"], ["woman","dead","bug","abdominal","editable","file","layer"]],
[["man","pigeon","puppy"], ["man","pigeon","puppy","alternating","reach","kickback","stand","four","six","pack"]],
[["woman","pigeon","puppy"], ["woman","pigeon","puppy","alternating","reach","kickback"]],
[["man","viparita","row","parallel","bar"], ["man","dip","parallel","bar","gym"]],
[["woman","viparita","row","parallel","bar"], ["woman","viparita","row"]],
[["man","resistance","strap","row"], ["man","standing","row","home","thin","resistance","strap"]],
[["woman","resistance","strap","row"], ["woman","resistance","strap","crab","stepping"]],
[["man","single","arm","dumbbell","row"], ["man","single","arm","folded","up","row","character","setu"]],
[["woman","single","arm","dumbbell","row"], ["woman","single","arm","dumbbell","overhead","shoulder","press","character","setu"]],
[["man","dumbbell","deadlift"], ["man","dumbbell","deadlift"]],
[["woman","dumbbell","deadlift"], ["woman","dumbbell","deadlift"]],
[["man","dumbbell","pullover"], ["man","dumbbell","pullover"]],
[["woman","dumbbell","pullover"], ["woman","dumbbell","pullover","foot","lift"]],
[["man","renegade","row","dumbbell"], ["man","standing","dumbbell","row"]],
[["woman","renegade","row","dumbbell"], ["woman","renegade","row","adho","puppy","tap","plank"]],
[["man","back","stretch"], ["man","standing","reach","up","back","rotation","stretch"]],
[["woman","back","stretch"], ["woman","sport","superman","stretch","four","back","ab","dumbbell","loss"]],
[["man","cat","cow","stretch"], ["man","yoga","cat","cow","stand","stretch"]],
[["woman","cat","cow","stretch"], ["woman","yoga","cat","cow","stretch"]],
[["man","sit","up"], ["man","sit","up","abdominal"]],
[["woman","sit","up"], ["woman","sit","up"]],
[["man","strength","abdominal","ab","prone","single","one","foot","lift"], ["man","strength","abdominal","ab","prone","single","one","foot","lift"]],
[["woman","strength","abdominal","ab","prone","single","one","foot","lift"], ["woman","strength","abdominal","ab","prone","single","one","foot","lift"]],
[["man","sit","up","dumbbell"], ["man","sit","up","abdominal"]],
[["woman","sit","up","dumbbell"], ["woman","sit","up"]],
[["man","plank","push","up"], ["man","plank","push","up","movement","stepping","down","ab"]],
[["woman","plank","push","up"], ["woman","perfect","straddle","plank","push","up","bar"]],
[["man","neck","stretch"], ["man","chair","sitting","neck","turn","head","rotation","rolling","left","right","healthy","activity","office","stretch"]],
[["woman","neck","stretch"], ["woman","neck","stretch"]],
[["man","calf","stretch"], ["man","resistance","strap","calf","stretch"]],
[["woman","calf","stretch"], ["woman","foam","skater","calf","stretch"]],
[["man","butterfly","stretch"], ["man","stretch","up"]],
[["woman","butterfly","stretch"], ["woman","butterfly","stretch"]],
[["man","lat","bend"], ["man","lat","bend","stretching","hand","hip","sport"]],
[["woman","lat","bend"], ["woman","ab","lat","bend","resistance","strap"]],
[["man","alternating","reverse","lunge"], ["man","alternating","lunge","jump"]],
[["woman","alternating","reverse","lunge"], ["woman","alternating","lunge","forward","lift"]],
[["man","calf","lift"], ["man","standing","calf","lift","assisted","machine"]],
[["woman","calf","lift"], ["woman","standing","calf","lift","dumbbell"]],
[["man","wall","sit"], ["man","wall","sit"]],
[["woman","wall","sit"], ["woman","wall","sit"]],
[["man","shoulder","tap"], ["man","modified","plank","shoulder","tap"]],
[["woman","shoulder","tap"], ["woman","modified","plank","shoulder","tap"]],
[["man","incline","push","up"], ["man","push","up"]],
[["woman","incline","push","up"], ["woman","incline","push","up"]],
[["man","tuck","jump"], ["man","tuck","jump","cardio"]],
[["woman","tuck","jump"], ["woman","knee","tuck","jump"]],
[["man","diaphragmatic","breathing"], ["man","sitting"]],
[["woman","diaphragmatic","breathing"], ["woman"]],
[["man","pelvic","tilt"], ["man","sitting"]],
[["woman","pelvic","tilt"], ["woman"]],
[["man","hundred"], ["man","sitting"]],
[["woman","hundred"], ["woman","pilate","hundred"]],
[["man","roll","up"], ["man","plank","up"]],
[["woman","roll","up"], ["woman","roll","up"]],
[["man","hip","lift","butt","bridge"], ["man","hip","lift","butt","bridge","curling","shoulder","reach","bodyweight"]],
[["woman","hip","lift","butt","bridge"], ["woman","hip","lift","butt","bridge"]],
[["man","plank","knee","tap"], ["man","knee","plank","reach","swimmer","stand"]],
[["woman","plank","knee","tap"], ["woman","knee","plank"]],
[["man","lat","plank","dip"], ["man","lat","plank","abdominal"]],
[["woman","lat","plank","dip"], ["woman","lat","plank"]],
[["man","dead","bug"], ["man","dead","bug","abdominal","editable","file","layer"]],
[["woman","dead","bug"], ["woman","dead","bug","abdominal","editable","file","layer"]],
[["man","foot","lift"], ["man","hanging","foot","lift","lat","abdominal"]],
[["woman","foot","lift"], ["woman","foot","lift","reach","clap"]],
[["man","teaser"], ["man","sitting"]],
[["man","single","arm","folded","up","row","character","setu"], ["man","single","arm","folded","up","row","character","setu"]],
[["woman","single","arm","folded","up","row","character","setu"], ["woman","single","arm","folded","up","row","character","setu"]],
[["man","dumbbell","folded","up","reverse","hand","row"], ["man","dumbbell","folded","up","reverse","hand","row"]],
[["woman","dumbbell","folded","up","reverse","hand","row"], ["man","dumbbell","folded","up","reverse","hand","row"]],
[["man","renegade","alternating","plank","commando","row"], ["man","renegade","alternating","plank","commando","row"]],
[["woman","renegade","alternating","plank","commando","row"], ["man","renegade","alternating","plank","commando","row"]],
[["man","sitting","dumbbell","hand","down","wrist","curl","forearm"], ["man","sitting","dumbbell","hand","down","wrist","curl","forearm"]],
[["woman","sitting","dumbbell","hand","down","wrist","curl","forearm"], ["man","sitting","dumbbell","hand","down","wrist","curl","forearm"]],
[["man","wrist","extension"], ["man","kneeling","wrist","forearm","stretch"]],
[["woman","wrist","extension"], ["woman","back","extension","bhujang","lift"]],
[["man","standing","dumbbell","calf","lift","character","setu"], ["man","standing","dumbbell","calf","lift","character","setu"]],
[["woman","standing","dumbbell","calf","lift","character","setu"], ["man","standing","dumbbell","calf","lift","character","setu"]],
[["man","jump","calf","lift","press","character","setu"], ["man","jump","calf","lift","press","character","setu"]],
[["woman","jump","calf","lift","press","character","setu"], ["man","jump","calf","lift","press","character","setu"]],
[["man","jog","stance"], ["man","high","knee","forward","lift","jog","stance"]],
[["woman","jog","stance"], ["woman","jog","stance"]]
]

################################
# 2) FONCTIONS DE SIMILARITÉ
################################
def jaccard_similarity(setA, setB) -> float:
    """Calcule la similarité de Jaccard = intersection / union, entre 0.0 et 1.0."""
    intersec = setA.intersection(setB)
    union_ = setA.union(setB)
    if len(union_) == 0:
        return 0.0
    return len(intersec) / len(union_)

def compute_similarity(listA: List[str], listB: List[str]) -> float:
    """
    Calcule une similarité brute (Jaccard).
    Applique un petit bonus de +0.05 si on détecte 'man' d'un côté et 'woman' de l'autre.
    """
    setA, setB = set(listA), set(listB)
    sim = jaccard_similarity(setA, setB)

    # Bonus man <-> woman (exemple)
    if ("man" in setA and "woman" in setB) or ("woman" in setA and "man" in setB):
        sim += 0.05

    # On borne la similarité entre 0 et 1
    return max(0.0, min(sim, 1.0))

##########################################
# 3) CRÉATION DE LA LISTE DE TRIPLETS
##########################################
def build_training_data(
    pairs: List[Tuple[List[str], List[str]]], 
    add_negative_samples: bool = True,
    negative_sample_count: int = 100
) -> List[Tuple[List[str], List[str], float]]:
    """
    Transforme la liste de paires (listA, listB) en triplets (listA, listB, similarity).
    - add_negative_samples : True pour générer des exemples négatifs aléatoires.
    - negative_sample_count : combien d'exemples négatifs on souhaite ajouter.
    """
    data_triplets = []

    # 1) Génération des triplets positifs / semi-positifs
    for (listA, listB) in pairs:
        sim = compute_similarity(listA, listB)
        data_triplets.append((listA, listB, sim))

    # 2) Ajout de paires négatives pour mieux couvrir l'éventail [0..1]
    if add_negative_samples:
        all_lists = [p[0] for p in pairs] + [p[1] for p in pairs]
        for _ in range(negative_sample_count):
            randA = random.choice(all_lists)
            randB = random.choice(all_lists)
            if randA != randB:
                sim_neg = compute_similarity(randA, randB)
                # Si la similarité brute n'est pas trop haute, on force un score faible
                if sim_neg < 0.5:
                    sim_neg = min(sim_neg, 0.2)  # On borne à 0.2 max pour en faire un "négatif"
                    data_triplets.append((randA, randB, sim_neg))

    return data_triplets

################################
# 4) EXÉCUTION & SAUVEGARDE
################################
if __name__ == "__main__":
    # Construit le dataset complet
    training_data = build_training_data(
        ENTIRE_DATASET,
        add_negative_samples=True,
        negative_sample_count=150  # Ajuste si besoin
    )

    # Affiche un aperçu des 10 premiers triplets
    print("Exemple (10 premiers triplets) :")
    for i in range(10):
        tokensA, tokensB, score = training_data[i]
        print(f"{i+1}) {tokensA} VS {tokensB} => similarité={score:.2f}")

    # Pour obtenir TOUTES les lignes directement en JSON, décommente le code suivant :
    timestamp = get_timestamp()
    filename = f"o3_generate_data_train_v1-{timestamp}.json"

    with open(f"generates/{filename}", "w", encoding="utf-8") as f:
      json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nFichier '{filename}' créé avec l'ensemble des triplets.")
    
    print(f"\nNombre total de triplets générés : {len(training_data)}")
