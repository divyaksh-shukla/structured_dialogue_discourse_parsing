from tqdm import tqdm
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransToolkit import IndicProcessor
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--model_name", type=str, default="ai4bharat/indictrans2-indic-en-dist-200M")
args.add_argument("--src_lang", type=str, default="hin_Deva")
args.add_argument("--tgt_lang", type=str, default="eng_Latn")
args.add_argument("--batch_size", type=int, default=8)
args.add_argument("--max_length", type=int, default=512)
args.add_argument("--input_file", type=str, default="convin_data/convin_complete_dataset_no_embedding.pkl")
args.add_argument("--output_file", type=str, default="convin_data/convin_complete_dataset_no_embedding_translated.pkl")
args.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
args = args.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

ip = IndicProcessor(inference=True)
src_lang, tgt_lang = "hin_Deva", "eng_Latn"

df = pd.read_pickle("convin_data/convin_complete_dataset_no_embedding.pkl")
# df = pd.read_pickle("convin_data/convin_complete_dataset.pkl")
df.reset_index(inplace=True)

input_sentences = df['machine_transcript'].dropna().tolist()
input_idx = df['machine_transcript'].dropna().index.tolist()

class InputSentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def collate_fn(self, batch):
        
        try:
            preprocessed_batch = ip.preprocess_batch(
                batch,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )


            # Tokenize the sentences and generate input encodings
            inputs = tokenizer(
                preprocessed_batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)
        except Exception as e:
            print(e)
            breakpoint()
            inputs = {
                "input_ids": torch.zeros((args.batch_size, 1)).to(DEVICE),
                "attention_mask": torch.zeros((args.batch_size, 1)).to(DEVICE),
            }
        return inputs

input_dataset = InputSentenceDataset(input_sentences, tokenizer)
# Dataloader with random sampler
input_dataloader = DataLoader(input_dataset, batch_size=args.batch_size, collate_fn=input_dataset.collate_fn)

# input_sentences = [
#     "जब मैं छोटा था, मैं हर रोज़ पार्क जाता था।",
#     "हमने पिछले सप्ताह एक नई फिल्म देखी जो कि बहुत प्रेरणादायक थी।",
#     "अगर तुम मुझे उस समय पास मिलते, तो हम बाहर खाना खाने चलते।",
#     "मेरे मित्र ने मुझे उसके जन्मदिन की पार्टी में बुलाया है, और मैं उसे एक तोहफा दूंगा।",
# ]


translated = []

model.to(DEVICE)

for inputs in tqdm(input_dataloader, ncols=120, desc=f"Translating from {src_lang} to {tgt_lang}"):
# Generate translations using the model
    try:
        with torch.no_grad():
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        for input_sentence, translation in zip(input_sentences, translations):
            # print(f"{src_lang}: {input_sentence}")
            # print(f"{tgt_lang}: {translation}")
            translated.append(translation)
    except Exception as e:
        print(e)
        breakpoint()
        translated.append([""]*args.batch_size)

df.loc[input_idx, 'english_transcript'] = translated

open("convin_data/convin_complete_dataset_translations.txt", "w").write("\n".join(translated))

print(df.sample(5))   

df.to_pickle("convin_data/convin_complete_dataset_no_embedding_translated.pkl")
        
# python translate_to_english.py --device cuda:0
