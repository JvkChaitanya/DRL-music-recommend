"""
Create a small subset of training data for quick model testing.
"""
import pickle
from pathlib import Path

def main():
    processed_path = Path("data/processed")
    subset_path = Path("data/subset")
    subset_path.mkdir(exist_ok=True)
    
    # Load raw sequences (large) - actually we can load the previously saved subset to save time if it exists
    # But let's load from processed to be safe.
    print("Loading train data...")
    with open(processed_path / 'train.pkl', 'rb') as f:
        train = pickle.load(f)
    train_subset = train[:10000]  # 10K samples
    del train
    
    print("Loading val data...")
    with open(processed_path / 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    val_subset = val[:2000]
    del val
    
    print("Loading test data...")
    with open(processed_path / 'test.pkl', 'rb') as f:
        test = pickle.load(f)
    test_subset = test[:2000]
    del test
    
    # --- Remap IDs ---
    print("Building new vocabulary for subset...")
    old_vocab = pickle.load(open(processed_path / 'vocab.pkl', 'rb'))
    
    active_items = set()
    for s in train_subset + val_subset + test_subset:
        active_items.update(s['sequence'])
        active_items.add(s['target'])
    
    if 0 in active_items: active_items.remove(0)
    
    sorted_active = sorted(list(active_items))
    print(f"Active items: {len(sorted_active):,} (was {len(old_vocab):,})")
    
    # New mapping: Old ID -> New ID
    # Start ID from 1 (0 is padding)
    id_map = {old_id: i+1 for i, old_id in enumerate(sorted_active)}
    id_map[0] = 0
    
    # Create new vocab list/dict
    # old_vocab is likely a dict or list. Let's assume it's a list/dict mapping ID->Name or Name->ID.
    # We'll create a new vocab mapping New_ID -> Name
    # Need to check structure of old_vocab. Assuming it's a dict {id: name} or {name: id}.
    # Let's verify by checking 1 item.
    # Actually, easiest is just to save the ID map if we need to look up names later.
    # For now, let's just save the new vocab size or dummy list.
    new_vocab = {new_id: f"item_{new_id}" for new_id in id_map.values()} 
    # Logic to try to preserve names if possible, but simplicity first.
    
    # Remap function
    def remap(dataset):
        remapped = []
        for sample in dataset:
            new_seq = [id_map.get(i, 0) for i in sample['sequence']]
            new_target = id_map.get(sample['target'], 0)
            if new_target != 0: # valid target
                 remapped.append({'user_id': sample['user_id'], 'sequence': new_seq, 'target': new_target})
        return remapped

    print("Remapping datasets...")
    train_remapped = remap(train_subset)
    val_remapped = remap(val_subset)
    test_remapped = remap(test_subset)
    
    # Save
    with open(subset_path / 'train.pkl', 'wb') as f:
        pickle.dump(train_remapped, f)
    with open(subset_path / 'val.pkl', 'wb') as f:
        pickle.dump(val_remapped, f)
    with open(subset_path / 'test.pkl', 'wb') as f:
        pickle.dump(test_remapped, f)
    with open(subset_path / 'vocab.pkl', 'wb') as f:
        pickle.dump(new_vocab, f)
        
    print(f"\n✓ Subset created in {subset_path}/")
    print(f"  Train: {len(train_remapped):,}")
    print(f"  Val: {len(val_remapped):,}")
    print(f"  Test: {len(test_remapped):,}")
    print(f"  Vocab: {len(new_vocab):,}")

if __name__ == "__main__":
    main()
