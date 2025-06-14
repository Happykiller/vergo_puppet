# convert_jsonl.py (version robuste)
import sys
import json

def convert_jsonl_to_json_streaming(input_path: str, output_path: str):
    """
    Convertit un fichier .jsonl en .json en utilisant le streaming.
    Idéal pour les très gros fichiers, car il a une empreinte mémoire très faible.
    """
    try:
        print(f"Streaming conversion from {input_path} to {output_path}...")
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            # Écrire le crochet d'ouverture du tableau JSON
            f_out.write('[\n')
            
            first_line = True
            for line in f_in:
                # Si ce n'est pas la première ligne, ajouter une virgule pour séparer les objets
                if not first_line:
                    f_out.write(',\n')
                
                # Valider que la ligne est un JSON valide (optionnel mais sûr)
                # puis la ré-écrire avec une indentation pour un joli formatage
                obj = json.loads(line)
                json.dump(obj, f_out, ensure_ascii=False, indent=2)

                first_line = False

            # Écrire le crochet de fermeture
            f_out.write('\n]\n')
            
        print("Conversion successful!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl.py <input_file.jsonl> <output_file.json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Utilisez la version streaming ici
    convert_jsonl_to_json_streaming(input_file, output_file)