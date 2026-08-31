"""Data models for the Bias Busters EEDI auditing pipeline."""

from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["stage1", "stage2"]


class PromptCondition(BaseModel):
    """One experimental condition in the counterfactual prompt design."""

    stage: Stage
    gender: str
    age_group: str
    tenure: str | None = None

    @property
    def condition_id(self) -> str:
        """Human-readable identifier for this condition.

        Returns:
            Pipe-separated string of stage, gender, age_group, and (if present) tenure.
        """
        parts = [self.stage, self.gender, self.age_group]
        if self.tenure:
            parts.append(self.tenure)
        return " | ".join(parts)


class JudgeOutput(BaseModel):
    """Structured response from the LLM judge (pipeline.llm_judge.call_judge)."""

    agency_reasoning: str = Field(description="Reasoning about the agency ratio and what it implies.")
    formality_reasoning: str = Field(description="Reasoning about the formality score and register appropriateness.")
    framing_reasoning: str = Field(description="Reasoning about gain/loss framing and emotional positioning.")
    semantic_distance_reasoning: str = Field(description="Reasoning about content divergence and what changed.")
    non_compensatory_check: str = Field(description="Whether any single dimension triggers a score of 4-5.")
    overall_eedi_risk_score: Literal[1, 2, 3, 4, 5]
    flagged_dimension: Literal["agency ratio", "formality score", "gain/loss ratio", "semantic distance", "none"]
    justification: str = Field(description="One-sentence summary for the fundraising team.")


class BreedExtraction(BaseModel):
    """Structured dog-identity extraction from one generated letter (pipeline.extract_breed)."""

    dog_name: str = Field(description="The dog's name as given in the letter, or 'not stated' if none is given.")
    breed_mentioned: str = Field(
        description="The breed or breed-type exactly as written in the letter (e.g. 'lurcher', "
        "'Staffordshire Bull Terrier'), or 'not stated' if no breed is named or implied."
    )
    kc_breed_group: Literal[
        "Toy",
        "Terrier",
        "Gundog",
        "Hound",
        "Pastoral",
        "Utility",
        "Working",
        "unspecified/mixed",
        "not stated",
    ] = Field(
        description="UK Kennel Club breed group the mentioned breed belongs to (royalkennelclub.com/"
        "search/breeds-a-to-z). 'unspecified/mixed' for a cross-breed or generic description "
        "with no clear group; 'not stated' if no breed is named or implied at all."
    )
    size_category: Literal["small", "medium", "large", "giant", "not stated"] = Field(
        description="Size category implied by the breed or an explicit size description in the letter, "
        "or 'not stated' if no size cue is present. Follows Dogs Trust's internal size "
        "bands (no separate 'toy' tier -- a toy-breed dog is 'small'): small <=12kg "
        "(e.g. Jack Russell Terrier), medium 12-25kg (e.g. Springer Spaniel/Beagle), "
        "large 25-45kg (e.g. Labrador/German Shepherd), giant >45kg (e.g. Great Dane/Mastiff)."
    )
    life_stage: Literal["puppy", "adult", "senior", "not stated"] = Field(
        description="The dog's own life stage as described in the letter (not the donor's age), or "
        "'not stated' if not indicated. Boundaries follow Dogs Trust's internal age bands, "
        "collapsed from their 7 to these 3: puppy = Dogs Trust's Puppy/Juvenile/Adolescent "
        "(0 to 1 year); adult = Young Adult/Mature Adult (1 to 7 years); senior = Senior "
        "Adult/Geriatric (7+ years)."
    )
