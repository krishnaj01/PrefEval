# PrefEval course project (Krishna Jhanwar, 12341250)

This is a fork of amazon-science/PrefEval (ICLR 2025). The project proposal is `12341250_KrishnaJhanwar_Proposal__PrefEval__ML_Project.pdf`
and the paper is `8673_Do_LLMs_Recognize_Your_Pr.pdf`. It runs in three phases:
1. **Phase 1, the current phase:** reproduce the paper's classification-task (MCQ) baselines with a **local** model. See `docs/PHASE1.md`.
2. Phase 2: train a small "does this turn state a preference?" classifier (DistilBERT/ModernBERT/MiniLM).
3. Phase 3: classifier → preference memory + modern retriever (bge/e5), evaluated against the Phase 1 baselines.

## Environment
- Conda env `prefeval` (Python 3.11): `source ~/anaconda3/etc/profile.d/conda.sh && conda activate prefeval`.
  Phase 1 deps are in `requirements-phase1.txt`. The original `requirements.txt` (vLLM, torch 2.4, boto3) is NOT used.
- Inference runs on **Ollama** (`localhost:11434`, systemd service), not vLLM and not Bedrock. The GPU is an RTX 3060 with 12 GB.
- Run everything from the repo root as modules: `python -m phase1.runner ...` and `python -m phase1.aggregate`.

## Layout
- `phase1/` holds the active Phase 1 pipeline (see `phase1/CLAUDE.md`).
- `configs/phase1/` holds `models.yaml` and one YAML per experiment tier.
- `results/phase1/` holds Phase 1 baseline results, which are committed. **Never write Phase 2/3 outputs here.**
- `logs/` holds verbose run logs and is git-ignored.
- `classification_task/`, `generation_task/`, `utils/`, `run_experiments.py`, `*.patch` are the upstream code plus
  the user's earlier (broken) local-vLLM attempt. They are **not used** by Phase 1. Do not modify or delete them without asking.

## Working rules (from the user)
- **Long-running jobs** (anything over ~5 min): give the user the command, a time estimate, and the tmux
  instructions (`scripts/phase1_tmux.sh <config>`). Do not run it yourself. Short smoke tests are fine to run.
- Everything must be resumable and save progressively. The resume unit is one question (an fsync'd JSONL line).
- Keep phases clearly separated: separate code packages, configs, and results trees per phase.
- Do not commit or push unless asked. Tell the user *when* a push is advisable: after code is verified, after each tier, and every 1–2 h during long tiers.
- Explain plans and flows transparently. The user wants to understand every step.

## Paper facts that are easy to get wrong
- Fig. 6 token rows 0.2k/1k/3k map to `inter_turns` 0/3/8, and 10k/16k/23k map to 28/48/68 (paper "turns" = inter_turns + 2).
- Llama3-8B has an 8k context, so only 0.2k–3k is possible. Mistral-7B-v0.2 reaches 23k.
- The paper's numbers are averaged over all 20 topics. The generation task needs a Claude-3-Sonnet judge, so it is not reproducible locally.
