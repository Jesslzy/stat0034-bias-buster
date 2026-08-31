# Bias Busters

**STAT0034 Research Project — UCL MSc Statistics**
Auditing LLM-generated donor engagement strategies for EEDI bias, in collaboration with Dogs Trust.

---

## Overview

Bias Busters is a counterfactual auditing pipeline for examining EEDI-relevant variation in AI-generated fundraising letters. It compares GPT, Gemini, and Claude while varying the recipient's gender, age group, and supporter tenure. Each letter is assessed using four linguistic metrics, a structured LLM-based EEDI risk score, and a separate extraction of the dog placed at the centre of the appeal. A stratified sample supports independent human validation of the automated judge.

### Experiment design

| Stage | Prompt attributes | Conditions | Outputs |
|-------|-------------------|-----------:|--------:|
| Stage 1 | Gender × age group | 35 | 525 |
| Stage 2 | Gender × age group × supporter tenure | 140 | 2,100 |

Each condition was generated five times by each of the three tools, producing 2,625 letters. Semantic distance uses the female, 55–64 condition as its Stage 1 reference and adds long-term high-value supporter for Stage 2.

---

## Project structure

```
stat0034-repo/
├── app.py                              # Streamlit UI (single-letter scoring)
├── main.py                             # CLI entry point (collect / score / extract-breed / judge / sample / validate)
├── constants.py                        # Shared domain constants, model IDs, retry/concurrency settings
├── requirements.txt
├── pyproject.toml                      # ruff config (lint/format)
├── .pre-commit-config.yaml             # pre-commit hooks (run manually -- see Dev tools below)
│
├── models/
│   └── models.py                       # Pydantic data models (PromptCondition, JudgeOutput, BreedExtraction)
│
├── pipeline/
│   ├── run_experiments.py              # `collect`      — Stage 1 & 2 data collection
│   ├── compute_metrics.py              # `score`         — linguistic metric computation
│   ├── extract_breed.py                # `extract-breed` — dog breed/size/life-stage extraction
│   ├── llm_judge.py                    # `judge`         — LLM judge (EEDI risk scoring)
│   ├── sample_for_annotation.py        # `sample`        — stratified sample for human validation
│   └── validate_annotations.py         # `validate`      — human vs. LLM judge agreement (Cohen's kappa)
│
├── metrics/
│   ├── agency.py                       # Agency ratio (Connotation Frames)
│   ├── formality.py                    # Formality score (Heylighen & Deacon F-score, via spaCy)
│   ├── framing.py                      # Gain/loss ratio (Harvard General Inquirer + domain extension)
│   └── semantic.py                     # Semantic distance (all-mpnet-base-v2)
│
├── prompt/
│   ├── experiments/
│   │   ├── stage1.txt                  # Stage 1 letter-generation prompt template
│   │   └── stage2.txt                  # Stage 2 letter-generation prompt template
│   └── agents/
│       ├── eedi_judge_system.txt       # Judge system prompt (rubric, anchor examples)
│       ├── eedi_judge_user.txt         # Judge user prompt template
│       ├── eedi_human_annotator_guide.md  # Rubric guide for human annotators (mirrors the judge prompt)
│       ├── breed_extraction_system.txt # Breed/size/life-stage extraction prompt
│       └── breed_extraction_user.txt   # Breed extraction user prompt template
│
├── resources/
│   ├── agency_power.csv                # Connotation Frames lexicon
│   └── inquireraugmentedBigDict1a.csv  # Harvard General Inquirer (augmented)
│
├── analysis/
│   ├── model_eedi.Rmd                  # Statistical modelling and diagnostics
│   └── dog_characteristics.py          # Descriptive dog-characteristic tables and figures
│
├── data/                                # Pipeline inputs, intermediate files, and outputs
│
└── utils/
    ├── llm_client.py                   # Shared async litellm call wrapper (retry/back-off)
    ├── logging_config.py
    └── prompt_loader.py
```

---

## Setup

### 1. Environment

Requires Python 3.11+.

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

> **Note:** `python -m spacy download en_core_web_sm` is equivalent but requires a stable connection to spaCy's servers. The direct pip install above is more reliable.

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
LITELLM_LOCAL_MODEL_COST_MAP=True

OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Lexicon resources

The two lexicon files must be downloaded manually and placed in `resources/`:

| File | Source |
|------|--------|
| `agency_power.csv` | Connotation Frames lexicon (Sap et al., 2017) — download `agency_power.csv` from maartensap.com/connotation-frames |
| `inquireraugmentedBigDict1a.csv` | Harvard General Inquirer — augmented spreadsheet from wjh.harvard.edu/~inquirer |

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Paste an AI-generated fundraising letter in the left box. A reference letter can be added in the right box to calculate semantic distance. Click **Analyse EEDI risk** to compute the linguistic metrics, extract dog characteristics, and run the LLM judge. The interface reports progress and presents the metric explanations alongside the final assessment.

### CLI pipeline

The six steps run in this order, with each command defaulting to the preceding step's output:

```bash
# 1. Collect LLM outputs across all Stage 1 & 2 conditions
python main.py collect --stage all --replicates 5 --out data/raw_outputs.csv

# 2. Compute the four linguistic metrics
python main.py score --in data/raw_outputs.csv --out data/scored_outputs.csv

# 3. Extract dog breed / size / life-stage info from each letter
python main.py extract-breed --in data/scored_outputs.csv --out data/scored_outputs_with_breed.csv

# 4. Run the LLM judge to assign EEDI risk scores
python main.py judge --in data/scored_outputs_with_breed.csv --out data/judged_outputs.csv

# 5. Draw a stratified sample for human inter-rater validation
python main.py sample --in data/judged_outputs.csv --out data/annotation_sample.csv

# 6. Once the annotator returns the completed workbook, compute agreement
python main.py validate --annotations data/annotation_sample_blind.xlsx --reference data/annotation_sample.csv
```

Every step supports `--help` for full options. Useful flags:

- `collect --resume` — skip rows already present in `--out` (resume an interrupted run). `--platforms` accepts one or more of `gpt gemini claude`.
- `score --only {agency,formality,framing,semantic}` — recompute just one metric on an already-scored file, leaving the other columns untouched (defaults to reading/writing `data/scored_outputs.csv` in place when set).
- `extract-breed --resume` and `judge --resume` — skip rows already processed in `--out`.
- `judge --limit N` — only judge the first N pending rows, for a quick test run before committing to a full pass.
- `judge --temperature` — omitted by default so reasoning-family judge models (e.g. gpt-5.x) use their own default rather than erroring on an explicit value; pass e.g. `--temperature 0` for deterministic scoring on a non-reasoning model.
- `sample --n` / `--seed` — sample size (default 150) and random seed (default 42).

The API-dependent stages use asynchronous requests with explicit concurrency limits. Results are handled as calls finish, allowing logging and checkpoint files to be updated without waiting for the slowest request in a batch. Shared DataFrames, counters, and output files are updated by the coordinating function rather than by concurrent worker functions.

---

## Metrics

| Metric | Method | Range |
|--------|--------|-------|
| **Agency ratio** | Connotation Frames verb lexicon (Sap et al., 2017) | −1 (low agency) → +1 (high agency) |
| **Formality score** | Heylighen & Deacon (1999) F-score via spaCy POS tags | 0 (informal) → 100 (formal) |
| **Gain/loss ratio** | Harvard General Inquirer lexicon (Stone et al., 1966) plus a hand-curated animal-welfare/fundraising domain extension (e.g. rescue, rehomed, euthanised, abandoned) | −1 (loss-framed) → +1 (gain-framed) |
| **Semantic distance** | Cosine distance via all-mpnet-base-v2 from the reference condition's output | 0 (identical) → 2 (opposite embedding directions) |

### EEDI risk score

The LLM judge (`openai/gpt-5.5-2026-04-23` by default; see `constants.DEFAULT_JUDGE_MODEL`) assigns a holistic score from 1 to 5. Its rubric defines EEDI principles, supplies score anchors, requests a short assessment of each linguistic measure, and applies a non-compensatory rule under which one severe concern can determine the overall score. The judge receives the complete letter and four metric values, but it is blind to the demographic condition, generating platform, and dog-characteristic extraction. A Pydantic schema (`JudgeOutput`) validates the score, flagged dimension, justification, and supporting assessments.

| Score | Meaning |
|-------|---------|
| 1 | No concern |
| 2 | Negligible |
| 3 | Review recommended |
| 4 | Revision recommended |
| 5 | Severe risk |

After scoring, `judge` logs Pearson and Spearman correlations between the EEDI score and a normalised composite of the four metrics. This diagnostic indicates how closely the judge's assessments align with a simple combination of the metric values.

### Breed / dog-identity extraction

`extract-breed` identifies the dog presented in each letter. This captures a potential channel of variation that the linguistic metrics do not measure: generated letters may use similar language while selecting different types of dogs for different recipient profiles. The structured `BreedExtraction` output contains five fields:

| Field | Values |
|-------|--------|
| `dog_name` | Name stated in the letter, or "not stated" |
| `breed_mentioned` | Breed or breed type stated in the letter, or "not stated" |
| `kc_breed_group` | UK Kennel Club's 7 official breed groups (Toy, Terrier, Gundog, Hound, Pastoral, Utility, Working), or "unspecified/mixed" / "not stated" |
| `size_category` | small / medium / large / giant / not stated — Dogs Trust's internal weight bands (small ≤12kg, medium 12–25kg, large 25–45kg, giant >45kg); no separate "toy" tier |
| `life_stage` | puppy / adult / senior / not stated (the dog's own life stage, not the donor's age) — collapsed from Dogs Trust's internal 7-band scheme: puppy 0–1yr, adult 1–7yr, senior 7yr+ |

Dog characteristics are analysed descriptively and remain separate from the EEDI risk score. Generate the summary tables and figures with:

```bash
python analysis/dog_characteristics.py \
    --input data/judged_outputs_final.csv \
    --output-dir analysis/dog_characteristics_output
```

The script reports breed-group, size, and life-stage distributions across demographic groups, supporter-tenure groups, and platforms. It produces separate stacked-bar figures for the demographic comparisons and the tenure/platform comparisons, together with CSV summaries and a difference heatmap.

### Human inter-rater validation

`sample` draws a blind-scoring sample from `judged_outputs.csv` for human annotators to independently score against the same rubric the LLM judge used, to validate whether the two agree. All five EEDI score categories (and all three platforms within each) are treated as equally important to validate, so the sample is split as evenly as possible across every `(eedi_score, platform)` cell — a thin cell (e.g. score 5, which is rare) simply contributes whatever exists rather than distorting the rest, so the realised sample size may come in a little under `--n`.

It writes three files:

| File | Purpose |
|------|---------|
| `annotation_sample.csv` | Full reference rows (all columns, including the judge's score) for later kappa computation |
| `annotation_sample_blind.csv` / `.xlsx` | Annotator-facing rows — only what the judge itself saw (`response_text` + the four metrics), plus an `item_id` to re-join against the full file afterwards. No demographic labels, platform, or judge score — annotators score blind |

The `.xlsx` version is a formatted, fillable workbook (wrapped text, frozen header, dropdown validation for the score and flagged-dimension columns) built for actually handing to an annotator; see `eedi_human_annotator_guide.md` for the scoring instructions to send alongside it.

Once the annotator returns the completed workbook, `validate` joins it back against `annotation_sample.csv` (via `item_id`) and reports a human-vs-judge confusion matrix, exact and within-one-point agreement rates, Spearman correlation, and both unweighted and quadratic-weighted Cohen's kappa (`sklearn.metrics.cohen_kappa_score`). Weighted kappa is the primary statistic, since the EEDI score is ordinal 1–5 and an off-by-one disagreement is far milder than an off-by-four one.

---

## Statistical analysis (R)

`analysis/model_eedi.Rmd` analyses the four linguistic measures in two stages. For each measure, the script first assesses random intercepts for prompt condition and platform within condition. It then applies backward elimination to the fixed effects while preserving required lower-order terms. The current analysis uses ordinary linear regression where the random-effect tests do not support retaining a random intercept. Final-model joint tests and follow-up comparisons are corrected within their stated test families using the Benjamini–Hochberg procedure.

The R Markdown also contains model diagnostics, comparisons of explanatory contribution, and a cumulative link mixed model for the ordinal EEDI risk score. The latter uses the GPT subset and separates each linguistic measure into within-condition and between-condition components. Dog-characteristic selection is not modelled in the R Markdown; it is examined descriptively with `analysis/dog_characteristics.py`.

Requires R and RStudio. Open `analysis/model_eedi.Rmd` and Knit. Its setup chunk lists the required packages (`readr`, `dplyr`, `lme4`, `lmerTest`, `ordinal`, `emmeans`, `performance`, `see`, and `broom.mixed`). The default data source is `data/judged_outputs_final.csv`.

---

## Dev tools

`ruff` (lint + format, config in `pyproject.toml`) and a handful of general hygiene checks are configured via `.pre-commit-config.yaml`, but the git hook is **not** installed automatically — run checks manually when you want them:

```bash
pip install pre-commit   # already in requirements.txt
pre-commit run --all
```

If you do want it to run automatically on every commit, `pre-commit install` sets that up (and `pre-commit uninstall` reverses it).
