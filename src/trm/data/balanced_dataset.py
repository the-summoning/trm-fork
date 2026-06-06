import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pathlib import Path

from trm.data.common import PuzzleDatasetMetadata


def build_balanced_dataset(
    dataset_path: str | Path,
    split: str = "train",
    set_name: str = "all",
    num_examples_per_puzzle: int = 3,
    seed: int = 42,
    max_samples: int | None = None
):
    rng = np.random.default_rng(seed)
    
    base_dir = Path(dataset_path)
    split_dir = base_dir / split

    inputs = np.load(split_dir / f"{set_name}__inputs.npy", mmap_mode="r")
    labels = np.load(split_dir / f"{set_name}__labels.npy", mmap_mode="r")
    puzzle_ids = np.load(split_dir / f"{set_name}__puzzle_identifiers.npy")
    puzzle_indices = np.load(split_dir / f"{set_name}__puzzle_indices.npy")

    with open(base_dir / "identifiers.json", "r") as f:
        id_to_name_mapping = json.load(f)
        
    with open(split_dir / "dataset.json", "r") as f:
        old_metadata_dict = json.load(f)

    real_groups = {}
    PuzzleIdSeparator = "|||"

    for p_idx, pid in enumerate(puzzle_ids):
        pid_int = int(pid)
        if pid_int == 0 or pid_int >= len(id_to_name_mapping):
            continue
            
        full_name = id_to_name_mapping[pid_int]
        original_name = full_name.split(PuzzleIdSeparator)[0]
        real_groups.setdefault(original_name, []).append(p_idx)

    print(f"Trovati {len(real_groups)} puzzle ORIGINALI unici basici.")

    selected_global_examples = []
    selected_puzzle_ids = []

    # 5. SELEZIONE: Itera sui 120 puzzle reali
    for original_name, p_indices_list in real_groups.items():
        p_indices_list = np.array(p_indices_list)
        
        # Scegliamo K varianti distinte di questo specifico puzzle originale
        k = min(num_examples_per_puzzle, len(p_indices_list))
        chosen_puzzles = rng.choice(p_indices_list, size=k, replace=False)
        
        for p_idx in chosen_puzzles:
            start_ex = puzzle_indices[p_idx]
            end_ex = puzzle_indices[p_idx + 1]
            
            # CORREZIONE: Prendiamo TUTTI gli esempi inclusi in questa variante, non solo il primo!
            for idx in range(start_ex, end_ex):
                selected_global_examples.append(idx)
                # Associamo l'ID di questa variante a ciascuno dei suoi esempi
                selected_puzzle_ids.append(puzzle_ids[p_idx])

    indices_to_extract = list(selected_global_examples)

    if max_samples is not None and max_samples < len(indices_to_extract):
        indices_to_extract = indices_to_extract[:max_samples]
        selected_puzzle_ids = selected_puzzle_ids[:max_samples]
        print(f"Dataset troncato a max_samples. Campioni estratti: {len(indices_to_extract)}")
    else:
        print(f"Numero di esempi unici selezionati finali: {len(indices_to_extract)}")

    total_selected_examples = len(indices_to_extract)
    
    final_inputs = torch.from_numpy(np.array([inputs[idx] for idx in indices_to_extract], dtype=np.int32))
    final_labels = torch.from_numpy(np.array([labels[idx] for idx in indices_to_extract], dtype=np.int32))
    final_ids = torch.tensor(selected_puzzle_ids, dtype=torch.int32)
    
    old_metadata_dict.update({
        "total_puzzles": len(real_groups), # Numero effettivo di puzzle unici estratti
        "total_groups": len(real_groups),  # Nel dataset bilanciato, ogni gruppo è un puzzle unico
        "mean_puzzle_examples": float(total_selected_examples / max(1, len(real_groups))) # Nuova media
    })
    
    metadata = PuzzleDatasetMetadata(**old_metadata_dict)

    return TensorDataset(final_inputs, final_labels, final_ids), metadata


def run_dataset_sanity_check(dataset_path: str, pca_dataset):
    print("\n" + "="*40)
    print("      AVVIO SANITY CHECK DATASET PCA      ")
    print("="*40)
    
    # 1. Carica il file identifiers.json originale per il controllo incrociato
    with open(os.path.join(dataset_path, "identifiers.json"), "r") as f:
        id_to_name_mapping = json.load(f)
        
    # Calcola quanti puzzle ARC base unici esistono nel mapping originale
    PuzzleIdSeparator = "|||"
    unique_base_puzzles = set()
    for name in id_to_name_mapping:
        if name != "<blank>":
            unique_base_puzzles.add(name.split(PuzzleIdSeparator)[0])
            
    num_base_puzzles = len(unique_base_puzzles)
    print(f"[1] Numero di puzzle ARC originali unici trovati nel JSON: {num_base_puzzles}")
    
    # 2. Controllo sulle dimensioni del TensorDataset generato
    inputs, labels, puzzle_ids = pca_dataset.tensors
    
    print(f"[2] Shape dei tensori estratti:")
    print(f"    - Inputs shape: {inputs.shape}")
    print(f"    - Labels shape: {labels.shape}")
    print(f"    - Puzzle IDs shape: {puzzle_ids.shape}")
    
    # Verifica teorica della dimensione delle righe
    expected_rows_max = num_base_puzzles * 3 # assumendo num_examples_per_puzzle=3
    print(f"    - Righe attese (al massimo): {expected_rows_max}")
    
    if inputs.shape[0] <= expected_rows_max and inputs.shape[0] > 0:
        print("    -> SUCESS: La dimensione delle righe è coerente e bilanciata!")
    else:
        print("    -> ERROR: La dimensione delle righe è anomala!")

    # 3. Controllo di Integrità sui Campioni (Prendiamo 3 indici casuali estratti)
    print(f"\n[3] Ispezione coerenza ID/Nomi su campioni casuali:")
    num_samples = len(puzzle_ids)
    
    # Scegliamo fino a 3 esempi estratti a caso dal nuovo dataset
    sample_indices = np.random.choice(range(num_samples), size=min(3, num_samples), replace=False)
    
    for idx in sample_indices:
        p_id_numeric = int(puzzle_ids[idx].item())
        
        # Risaliamo al nome registrato nel JSON usando l'ID numerico che viaggerà nel modello
        if p_id_numeric < len(id_to_name_mapping):
            resolved_name = id_to_name_mapping[p_id_numeric]
            base_name = resolved_name.split(PuzzleIdSeparator)[0]
            
            print(f"    - Esempio estratto n°{idx}:")
            print(f"      * ID Numerico passato al modello: {p_id_numeric}")
            print(f"      * Nome risolto da JSON:          {resolved_name}")
            print(f"      * Radice Puzzle originale (ARC): {base_name}")
        else:
            print(f"    - ERROR: L'ID numerico {p_id_numeric} è fuori dal range del dizionario JSON!")

    # 4. Verifica finale che non ci siano ID pari a 0 (<blank>) nei dati selezionati
    zeros_found = torch.sum(puzzle_ids == 0).item()
    print(f"\n[4] Controllo record di padding (<blank>):")
    if zeros_found == 0:
        print("    -> SUCCESS: Nessun ID di padding (0) rilevato nel dataset PCA.")
    else:
        print(f"    -> WARNING: Trovati {zeros_found} ID di padding pari a 0!")
        
    print("="*40)
    print("           FINE SANITY CHECK            ")
    print("="*40 + "\n")


# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    # Sostituisci con il tuo path effettivo
    DATASET_PATH = "/home/davide/Scaricati/arc2concept-aug-1000" 
    
    dataset, metadata = build_balanced_dataset(
        dataset_path=DATASET_PATH, 
        split="test", 
        set_name="all", 
        num_examples_per_puzzle=3 # 3 esempi ben distribuiti per ogni puzzle originale
    )
    
    run_dataset_sanity_check(DATASET_PATH, dataset)
