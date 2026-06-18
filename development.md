# Development Log: AfriQA Hackathon

This document tracks our core experiments and architectural shifts throughout the hackathon.

## Timeline of Core Experiments

### Baseline & Early Experiments
- Baseline models explored: Various zero-shot / generic models.
- Started with generic retrieval or RAG baseline yielding lower scores.
- **Score:** ~0.5898 (TF-IDF Baseline)

### Experiment A: The Gemma "Flip the Ratio" Shift
- **Change:** Moved from mostly retrieval (~99%) to predominantly generation (~80%).
- **Mechanism:** Implemented `run_gemma_vllm.py` using Gemma-4-31B-it via vLLM. 
- **Retrieval Threshold:** Set a uniform dense retrieval (E5) cosine similarity threshold of `0.96`.
- **System Prompt:** Added a robust 6-point instructional prompt enforcing strict language-matching, terminology echo, and dynamic output length constraints.
- **Dynamic Length Constraints:** Used the median length of the top retrieved examples to limit `max_tokens` (with a 40% headroom).
- **Result:** ROUGE jumped dramatically. Public score reached **0.6074**.

### Experiment B: Per-Language Threshold Calibration (p90)
- **Change:** Calibrated retrieval thresholds individually for every language subset to guarantee a perfect 90/10 split between Generation/Retrieval.
- **Diagnostic Finding:** `Eng_Eth` had a median E5 similarity of `0.994`! The flat `0.96` threshold was essentially invisible for it.
- **Mechanism:** Wrote `calibrate.py` to evaluate the in-sample distribution of similarities. 
- **Thresholds Applied:**
  - Aka_Gha: `0.979`
  - Amh_Eth: `0.988`
  - Eng_Eth: `0.997` (pulled slightly back from `0.999` for duplicate safety)
  - Eng_Gha: `0.966`
  - Eng_Ken: `0.989`
  - Eng_Uga: `0.995`
  - Lug_Uga: `0.983`
  - Swa_Ken: `0.988`
- **Result:** Reached Public Score **0.6264** (Rank 96).
  - LLM Judge: `0.8052` (Exceptional, medically accurate)
  - Rouge 1 F1: `0.6032` (Lacking exact lexical overlap vs the Top 1 competitor `0.7201`)

### Experiment D: Reciprocal Rank Fusion (RRF) Hybrid Retrieval
- **Change:** Implemented a Two-Stage Hybrid (Dense + Sparse) strategy for few-shot selection.
- **Mechanism:**
  1. **Stage 1 (Routing):** Fetch the Top 25 candidates using Dense E5. We still use *pure Dense* `top1_score` against the calibrated p90 thresholds to decide whether to retrieve exactly or generate.
  2. **Stage 2 (RRF Few-Shot Selection):** If routing to generation, we compute TF-IDF (character n-grams 3-5 with sublinear scaling) for those 25 candidates. We then fuse the Dense Ranks and Sparse Ranks using Reciprocal Rank Fusion (`k=60`).
- **Rationale:** African agglutinative morphology (e.g. Swahili, Luganda) means semantic vectors (E5) sometimes miss crucial lexical overlap and prefixes/suffixes. By using RRF with character-level TF-IDF, we guarantee Gemma is prompted with few-shot examples that possess the exact vocabulary of the query, directly targeting the ROUGE score deficit.
- **Other Changes:** Updated output script to append timestamps (`submission_YYYYMMDD_HHMMSS.csv`) to prevent accidental overwrites.
- **Status:** **COMPLETED**
- **Result:** Public score slightly decreased to **0.6245** (from 0.6264).
  - LLM Judge: `0.8056` (slight improvement, still exceptional)
  - Rouge 1 F1: `0.6016` (slight drop)
  - **Analysis:** RRF may have introduced too much noise by allowing TF-IDF to overrule Dense semantic relevance for the few-shot examples, or the prompt/length constraints are not fully capitalizing on the lexical alignment. We need to follow Opus's next steps: prompt refinement and sharpening length calibration.
