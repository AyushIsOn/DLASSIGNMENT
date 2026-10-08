# AcharyaGPT paper (draft, Elsevier `elsarticle` format)

- `main.tex`: manuscript; `refs.bib`: references; `figures/`: figures from the round-1 run
  (`results-round1` branch) and `docs/pipeline_flow.png`; `main.pdf`: compiled draft.
- Build: `tectonic main.tex` or `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
  It also compiles on Overleaf (upload the folder, choose pdfLaTeX).
- Fill in the red `[TODO: ...]` items: affiliation, generative-AI declaration, acknowledgements,
  target journal (`\journal{...}`).
- Every number comes from `results/round1/eval/results.json`, `generations.jsonl` and
  `training_summary.json` on the `results-round1` branch.
