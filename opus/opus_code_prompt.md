# Opus Code Generation Prompt

Hello Opus,

Based on your incredible strategy to transition from TF-IDF generation to Dense Retrieval + Per-Subset Routing, I am ready for you to rewrite our codebase. 

I have attached the two files you need to modify:
1. `run_gemma_vllm.py`
2. `launch_gemma.sbatch`

Please rewrite these files entirely to implement your strategy. Also, provide the code for the new `validate_local.py`.

### Critical Constraints to Include in Your Code
So we don't waste time debugging, please strictly ensure your generated code adheres to these constraints from our previous context:

1. **Offline Inference (No Internet):** 
   - The Slurm cluster has ZERO internet access during the job. 
   - We will pre-download `multilingual-e5-large` to the shared drive. In `run_gemma_vllm.py`, you must hardcode the path: `EMBED_MODEL_PATH = "/mnt/data/sashishj/models/intfloat_multilingual-e5-large"`. Do not use the `intfloat/multilingual-e5-large` string directly in `SentenceTransformer()`, use the local path so it doesn't try to connect to HuggingFace.
2. **Gemma API Endpoint:**
   - We run Gemma-4-31B offline via a local vLLM server inside a Singularity container. Ensure your `query_vllm` function continues to hit `http://localhost:8000/v1` and uses the OpenAI python client exactly as it does currently.
3. **Zindi Submission Format:**
   - The final output CSV must exactly have columns: `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM`.
   - The exact same predicted answer must be written to all three target columns for each ID. Do not drop any `NaN` values.
4. **Slurm & Cluster Variables:**
   - In `launch_gemma.sbatch`, we need `HSA_FORCE_FINE_GRAIN_PCIE=1`, `NCCL_SOCKET_IFNAME=lo`, and `export LD_LIBRARY_PATH=...` exactly as they are currently written to bypass the GLIBC issues. Do not remove our SLURM configurations or the Singularity `vllm_rocm.sif` command.

### Output Request
Since I am using an automated coding agent on my end, please provide the complete, finalized code for all files. If your platform supports generating a `.zip` file of the new scripts, please output a downloadable zip file! If not, just provide the full file contents in clean markdown blocks with the filename at the top of each block so my agent can parse them easily.
