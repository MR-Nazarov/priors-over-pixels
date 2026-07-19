# Priors Over Pixels: Present-Bias in Organ-Presence Grounding for Medical VLMs

Code and result data for the paper *"Priors Over Pixels: Present-Bias in
Organ-Presence Grounding for Medical VLMs"*, accepted to the **MICCAI 2026 SAFER**
workshop.

We test whether medical vision–language models answer organ-presence questions on
abdominal CT from the **image** or from **anatomical priors**. Using a POPE-style
negative-sampling protocol on the BTCV dataset, we find a systematic **present-bias**:
models affirm absent organs, the bias strengthens with model scale, persists without
the image, concentrates in large high-prior organs, and — via a controlled base-vs-
medical ablation (Gemma3 vs MedGemma) — is shown to be **intrinsic to the base VLM**,
mitigated (not caused) by medical adaptation.

## Repository layout

```
scripts/
  anatomy_prior_probe.py     # core harness: prompts (build_prompt), MedGemma wrapper,
                             #   slice rendering (W400/L40), slice sampling, hysteresis GT
  pope_negative_sampling.py  # POPE random/popular/adversarial samplers + query assembly
  pope_replay.py             # replay the exact persisted items through any model
  pope_replay_models.py      # Qwen2.5-VL and LLaVA-Med wrappers
  pope_abstention.py         # 14-cue abstention detector + confident-only specificity
  pope_bootstrap_cis.py      # case-clustered bootstrap CIs (seed 0, 10k) for the tables
  adv_zdist_analysis.py      # z-distance-to-nearest-presence logistic analysis
  pope_noimg_delta.py        # no-image control (text-only / blank-gray) deltas
  pope_render_control.py     # native triple-window render control (encoding invariance)
  gemma3_pope.py             # base-Gemma3 ablation runner (stage1 parse gate, stage2 full)
  gemma3_stage3.py           # base-vs-medical comparison (--tier 4b|27b)
  abstention_llm_annotator.py# LLM-annotator validation of the abstention detector
  fig_*.py, concept_slice_fig.py  # paper figures

results/medgemma_pope_negsampling/
  pope_results.json               # MedGemma-27B (main run)
  pope_results_medgemma4b.json    # MedGemma-4B
  pope_results_qwen.json          # Qwen2.5-VL-7B
  pope_results_llavamed.json      # LLaVA-Med
  gemma3_4b_pope.json             # Gemma3-4B (base ablation)
  gemma3_27b_pope.json            # Gemma3-27B (base ablation)
  pope_*summary*.json, pope_abstention.json, adv_zdist_*.json  # derived summaries

abstention_validation_sample_Maya.csv  # 100 responses + human abstention labels
```

Each result JSON is `{"records": [...]}`; a record is one `(sample_id, organ, format)`
verdict with `neg_strategy`, `ground_truth_present`, `parsed_present`, `parse_fail`,
and the verbatim model `raw` output.

## Result data (Hugging Face)

The per-model result JSONs are hosted as a Hugging Face dataset (kept out of git to
keep clones lean; no bandwidth caps):

**https://huggingface.co/datasets/Lexer1/priors-over-pixels-data**

Download them into place before running the analysis:

```
pip install huggingface_hub
hf download Lexer1/priors-over-pixels-data --repo-type dataset --local-dir .
# -> populates results/medgemma_pope_negsampling/*.json
```

## Environment

```
conda create -n pop python=3.12 && conda activate pop
pip install torch transformers nibabel numpy pillow pandas
```
Reproducing the **tables/CIs** needs only the provided result JSONs (no GPU, no data
download). The **z-distance analysis and figures** additionally need the BTCV label
volumes under `data/btcv/RawData/Training/{img,label}` (download BTCV separately; not
redistributed here). Re-running the **models** needs the gated checkpoints
(`google/medgemma-{4b,27b}-it`, `google/gemma-3-{4b,27b}-it`) and a GPU.

## Reproduce each table/figure (from the provided result JSONs — no GPU)

| Paper element | Command |
|---|---|
| POPE specificity/sensitivity + CIs (MedGemma-27B) | `python scripts/pope_bootstrap_cis.py` |
| Abstention table (binary + confident spec + CIs) | `python scripts/pope_bootstrap_cis.py` |
| Cross-model free-text specificity (all models) | `python scripts/pope_abstention.py` |
| Base-vs-medical ablation (Gemma3 vs MedGemma) | `python scripts/gemma3_stage3.py --tier 4b` / `--tier 27b` |
| z-distance slopes + near/far/whole-volume | `python scripts/adv_zdist_analysis.py` *(needs BTCV labels)* |
| No-image control | `python scripts/pope_noimg_delta.py` |
| Figures | `python scripts/fig_pope_collapse.py`, `fig_zdist.py`, `fig_fabrication.py` *(labels)* |
| Abstention-detector LLM validation | `python scripts/abstention_llm_annotator.py --backend api` *(ANTHROPIC_API_KEY)* |

## Re-run a model from scratch (needs GPU + gated checkpoints)

```
python scripts/pope_negative_sampling.py --run    # main MedGemma-27B sweep
python scripts/pope_replay.py --model qwen        # replay persisted items, another model
python scripts/gemma3_pope.py --model gemma3_4b --stage1   # parse gate
python scripts/gemma3_pope.py --model gemma3_4b --stage2   # full run
```

## Protocol summary

- **Hysteresis ground truth:** organ *present* at ≥50 voxels on a slice, clean
  *true-negative* at <5 voxels; 5–49 excluded from querying.
- **Negative sampling:** random (uniform absent), popular (highest marginal
  frequency), adversarial (highest slice-level co-occurrence with present focal
  organs; near-ubiquitous vessels excluded as conditioners).
- **Three output formats:** free-text, verdict-first JSON, reasoning-first JSON.
- **CIs:** patient-clustered bootstrap (resample the 30 BTCV cases, 10k reps,
  percentile, fixed seed 0).

## Vision encoders (per model cards)

All four models share the SigLIP-So400m/14 @896 encoder **architecture**, but the
**weights differ**: MedGemma-4B/27B use **MedSigLIP** (medically domain-adapted
SigLIP); Gemma3-4B/27B use the **general SigLIP**. The base-vs-medical pairs are
therefore scale- and architecture-matched but not encoder-weight-matched.

## Citation

```bibtex
@inproceedings{priorsoverpixels2026,
  title     = {Priors Over Pixels: Present-Bias in Organ-Presence Grounding for Medical VLMs},
  booktitle = {MICCAI 2026 Workshop on Safe and Fair Evaluation of Robustness (SAFER)},
  year      = {2026}
}
```

## License / data

Code released under the MIT License (see `LICENSE`). BTCV is not redistributed here
(download under its own terms). Model checkpoints are gated on Hugging Face under
their respective licenses. `abstention_validation_sample_Maya.csv` contains model
outputs and manual labels over public BTCV case identifiers — no patient data.
