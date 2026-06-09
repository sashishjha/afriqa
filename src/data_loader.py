"""Data loading, subset management, HuggingFace Dataset creation."""

import pandas as pd
from datasets import Dataset as HFDataset

SUBSETS = [
    "Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth",
    "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth",
]

SUBSET_TO_LANG = {
    "Eng_Uga": "English", "Aka_Gha": "Akan",  "Eng_Gha": "English",
    "Eng_Eth": "English", "Lug_Uga": "Luganda","Eng_Ken": "English",
    "Swa_Ken": "Swahili", "Amh_Eth": "Amharic",
}

SUBSET_TO_NLLB = {
    "Eng_Uga": "eng_Latn", "Aka_Gha": "aka_Latn", "Eng_Gha": "eng_Latn",
    "Eng_Eth": "eng_Latn", "Lug_Uga": "lug_Latn", "Eng_Ken": "eng_Latn",
    "Swa_Ken": "swh_Latn", "Amh_Eth": "amh_Ethi",
}


def load_data(path: str) -> pd.DataFrame:
    from pathlib import Path

    candidate = Path(path)
    if not candidate.is_file():
        alt = Path(__file__).resolve().parents[1] / candidate.name
        if alt.is_file():
            candidate = alt
        else:
            raise FileNotFoundError(
                f"Could not find dataset file: {path}. Tried {candidate} and {alt}."
            )

    df = pd.read_csv(candidate)
    if "output" in df.columns:
        df["output"] = df["output"].fillna("")
    df["input"] = df["input"].fillna("")
    return df


def filter_by_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    return df[df["subset"] == subset].reset_index(drop=True)


def filter_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    subs = [s for s, l in SUBSET_TO_LANG.items() if l == language]
    return df[df["subset"].isin(subs)].reset_index(drop=True)


def get_subset_distribution(df: pd.DataFrame) -> dict:
    return df["subset"].value_counts().to_dict()


# ── HuggingFace dataset factories ──────────────────────────────────────

def create_seq2seq_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 512,
    is_test: bool = False,
    input_prefix: str = "answer_question: ",
):
    """Create a tokenised HF Dataset for Seq2SeqTrainer."""

    cols_src = ["input"] if is_test else ["input", "output"]
    hf = HFDataset.from_pandas(df[cols_src].copy(), preserve_index=False)

    def _tok(batch):
        inputs = [f"{input_prefix}{q}" for q in batch["input"]]
        enc = tokenizer(
            inputs, max_length=max_input_length,
            truncation=True, padding="max_length",
        )
        if not is_test:
            lab = tokenizer(
                text_target=batch["output"],
                max_length=max_target_length,
                truncation=True, padding="max_length",
            )
            lab_ids = [
                [(t if t != tokenizer.pad_token_id else -100) for t in seq]
                for seq in lab["input_ids"]
            ]
            enc["labels"] = lab_ids
        return enc

    return hf.map(_tok, batched=True, remove_columns=hf.column_names, num_proc=1)


def create_causal_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 768,
    is_test: bool = False,
):
    """Create a tokenised HF Dataset for causal-LM training."""

    cols_src = ["input"] if is_test else ["input", "output"]
    hf = HFDataset.from_pandas(df[cols_src].copy(), preserve_index=False)

    def _tok(batch):
        all_ids, all_mask, all_labels = [], [], []
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"  # must be right-padded during training for correct label alignment
        for i in range(len(batch["input"])):
            prompt = f"Question: {batch['input'][i]}\nAnswer:"
            if not is_test:
                full = f"{prompt} {batch['output'][i]}{tokenizer.eos_token}"
                p_enc = tokenizer(prompt, add_special_tokens=False)
                f_enc = tokenizer(
                    full, max_length=max_length,
                    truncation=True, padding="max_length",
                )
                labels = list(f_enc["input_ids"])
                plen = min(len(p_enc["input_ids"]), len(labels))
                for j in range(plen):
                    labels[j] = -100
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                all_ids.append(f_enc["input_ids"])
                all_mask.append(f_enc["attention_mask"])
                all_labels.append(labels)
            else:
                enc = tokenizer(
                    prompt, max_length=max_length,
                    truncation=True, padding="max_length",
                )
                all_ids.append(enc["input_ids"])
                all_mask.append(enc["attention_mask"])
        tokenizer.padding_side = orig_padding_side
        out = {"input_ids": all_ids, "attention_mask": all_mask}
        if not is_test:
            out["labels"] = all_labels
        return out

    return hf.map(_tok, batched=True, remove_columns=hf.column_names, num_proc=1)
