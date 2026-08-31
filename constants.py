"""Domain constants for the Bias Busters EEDI auditing pipeline."""

GENDERS = ["female", "male", "non-binary", "term not listed", "prefer not to say"]
AGE_GROUPS = ["24 and under", "25-34", "35-44", "45-54", "55-64", "65+", "prefer not to say"]
TENURE_PROFILES = [
    "long-term high-value supporter",
    "seasonal (Christmas-only) supporter",
    "infrequent low-value contributor",
    "new prospect",
]

# Reference condition used as the baseline for semantic-distance scoring and
# for computing reference metric means. Shared by pipeline.compute_metrics
# and pipeline.llm_judge so the two stay in sync.
REFERENCE_GENDER = "female"
REFERENCE_AGE = "55-64"
REFERENCE_TENURE = "long-term high-value supporter"

PLATFORMS: dict[str, str] = {
    "gpt": "openai/gpt-5.4-mini-2026-03-17",
    "gemini": "gemini/gemini-3.1-flash-lite",
    "claude": "anthropic/claude-haiku-4-5-20251001",
}

# Judge model
DEFAULT_JUDGE_MODEL = "openai/gpt-5.5-2026-04-23"

# Shared retry policy for every litellm call (utils.llm_client).
RETRY_ATTEMPTS = 5

# Max concurrent in-flight requests per platform during `collect`.
CONCURRENCY_PER_PLATFORM = 5

# Max concurrent in-flight requests during `judge`.
JUDGE_CONCURRENCY = 10

# How often (in completed rows) `judge` checkpoints its progress to disk.
JUDGE_CHECKPOINT_EVERY = 50

# Model used for breed/dog-identity extraction (pipeline.extract_breed)
DEFAULT_BREED_MODEL = PLATFORMS["gpt"]

# Max concurrent in-flight requests during `extract-breed`.
BREED_CONCURRENCY = 4

# How often (in completed rows) `extract-breed` checkpoints its progress to disk.
BREED_CHECKPOINT_EVERY = 50
