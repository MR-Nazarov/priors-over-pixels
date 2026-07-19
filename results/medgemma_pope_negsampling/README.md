# Result data

The per-model result JSONs are **not stored in git** — they live on Hugging Face:

**https://huggingface.co/datasets/Lexer1/priors-over-pixels-data**

Download them into this directory before running the analysis scripts:

```
hf download Lexer1/priors-over-pixels-data --repo-type dataset --local-dir <repo-root>
```

This populates `results/medgemma_pope_negsampling/` with:
`pope_results.json` (MedGemma-27B), `pope_results_medgemma4b.json`,
`pope_results_qwen.json`, `pope_results_llavamed.json`, `gemma3_4b_pope.json`,
`gemma3_27b_pope.json`, and the derived summaries
(`pope_summary_allmodels.json`, `pope_abstention.json`, `adv_zdist_*.json`,
`pope_noimg_summary.json`).
