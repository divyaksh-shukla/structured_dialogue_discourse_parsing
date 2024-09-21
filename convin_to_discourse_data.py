import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_discourse_data(df, domains):
    discourse_data = []
    for domain in domains:
        domain_df = df[df['domain'] == domain]
        batches = domain_df['domain_batch'].unique()
        for batch in batches:
            batch_df = domain_df[domain_df['domain_batch'] == batch]
            dialogue_ids = batch_df['dialogue_id'].unique()
            for dialogue_id in dialogue_ids:
                dialogue_df = batch_df[batch_df['dialogue_id'] == dialogue_id]
                if len(dialogue_df) > 1:
                    discourse_data.append({
                        'id': f"{domain}_{batch}_{dialogue_id}",
                        "edus": [{"speaker": str(i%2), "text": x} for i, x in enumerate(dialogue_df['english_transcript'].tolist())],
                        "relations": []
                    })
    return discourse_data

def save_discourse_data(discourse_data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(discourse_data, open(path, "w", encoding="latin_1"), indent=2)

df = pd.read_pickle("convin_data/convin_complete_dataset_no_embedding_translated.pkl")
df.reset_index(inplace=True)

convin_discourse_train_path = Path("convin_data/train/train.json")
convin_discourse_test_path = Path("convin_data/test/test.json")
convin_discourse_dev_path = Path("convin_data/dev/dev.json")

domains = df['domain'].unique()
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, dev_df = train_test_split(train_df, test_size=0.08, random_state=42)

train_discourse_data = create_discourse_data(train_df, domains)
test_discourse_data = create_discourse_data(test_df, domains)
dev_discourse_data = create_discourse_data(dev_df, domains)

save_discourse_data(train_discourse_data, convin_discourse_train_path)
save_discourse_data(test_discourse_data, convin_discourse_test_path)
save_discourse_data(dev_discourse_data, convin_discourse_dev_path)

print(f"Train: {len(train_discourse_data)}")
print(f"Test: {len(test_discourse_data)}")
print(f"Dev: {len(dev_discourse_data)}")


# python convin_to_discourse_data.py
