"""Provide the Streamlit interface for EEDI screening."""

import asyncio
import base64
import html
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from constants import DEFAULT_BREED_MODEL, DEFAULT_JUDGE_MODEL
from metrics.agency import agency_ratio
from metrics.framing import gain_loss_ratio
from models.models import BreedExtraction
from pipeline.extract_breed import extract_breed_from_text
from pipeline.llm_judge import call_judge
from utils.logging_config import setup_logging
from utils.prompt_loader import load_prompt

setup_logging()
logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).parent


@st.cache_resource
def _get_judge_prompts() -> tuple[str, str]:
    """Load and cache judge prompts for the lifetime of the app process.

    Returns:
        The system prompt and user prompt template.
    """
    system = load_prompt("eedi_judge_system", subdir="agents")
    user = load_prompt("eedi_judge_user", subdir="agents")
    return system, user


@st.cache_resource
def _get_breed_prompts() -> tuple[str, str]:
    """Load and cache dog-extraction prompts for the app process.

    Returns:
        The system prompt and user prompt template.
    """
    system = load_prompt("breed_extraction_system", subdir="agents")
    user = load_prompt("breed_extraction_user", subdir="agents")
    return system, user


def _report_status(status, message: str) -> None:
    """Log a progress message and display it when a status handler is available.

    Args:
        status: Optional Streamlit status handler.
        message: Progress message to display.
    """
    logger.info(message)
    if status is not None:
        status.write(message)


def _score_text(response_text: str, reference_text: str | None, status=None) -> dict:
    """Compute linguistic metrics for the response text.

    Args:
        response_text: Generated fundraising text to assess.
        reference_text: Optional reference text for semantic distance.
        status: Optional Streamlit status handler.

    Returns:
        Computed linguistic metric values.
    """

    _report_status(status, "Analysing agency...")
    agency = agency_ratio(response_text)

    _report_status(status, "Analysing formality...")
    # Deferred: these two pull in spaCy and sentence-transformers/torch, which
    # take several seconds to import — only pay that cost once scoring is used.
    from metrics.formality import formality_score

    formality = formality_score(response_text)

    _report_status(status, "Analysing persuasive framing...")
    framing = gain_loss_ratio(response_text)

    scores = {
        "agency_ratio": agency,
        "formality_score": formality,
        "gain_loss_ratio": framing,
        "semantic_distance": None,
    }

    if reference_text:
        _report_status(status, "Comparing semantic content...")
        from metrics.semantic import compute_semantic_distances

        scores["semantic_distance"] = compute_semantic_distances([response_text], reference_text=reference_text)[0]

    return scores


def _eedi_badge(score: int | None) -> tuple[str, str]:
    """Return a label and Streamlit colour for an EEDI risk score.

    Args:
        score: EEDI risk score, or ``None`` when unavailable.

    Returns:
        The display label and colour for the score.
    """
    if score is None:
        return "Assessment unavailable", "grey"
    return {
        1: ("No concern", "green"),
        2: ("Negligible concern", "green"),
        3: ("Review recommended", "orange"),
        4: ("Revision recommended", "red"),
        5: ("Severe risk", "red"),
    }[score]


def _extract_dog_details(response_text: str, status=None) -> BreedExtraction | None:
    """Extract dog identity, breed group, size, and life stage from one text.

    Args:
        response_text: Generated fundraising text to assess.
        status: Optional Streamlit status handler.

    Returns:
        Structured dog details, or ``None`` if extraction fails.
    """
    message = "Extracting dog details..."
    logger.info(message)
    if status is not None:
        status.write(message)

    system_prompt, user_template = _get_breed_prompts()
    return asyncio.run(
        extract_breed_from_text(
            response_text=response_text,
            model=DEFAULT_BREED_MODEL,
            system_prompt=system_prompt,
            user_template=user_template,
        )
    )


def _run_judge(response_text: str, scores: dict, status=None) -> tuple[int | None, str | None, str | None]:
    """Call the LLM judge and return its structured response.

    Args:
        response_text: Generated fundraising text to assess.
        scores: Pre-computed linguistic metric values.
        status: Optional Streamlit status handler.

    Returns:
        The risk score, flagged area, and justification.
    """

    _report_status(status, "Preparing the EEDI assessment...")
    system_prompt, user_template = _get_judge_prompts()
    sem = scores["semantic_distance"]
    user_prompt = user_template.format(
        response_text=response_text,
        agency_ratio=scores["agency_ratio"],
        formality_score=scores["formality_score"],
        gain_loss_ratio=scores["gain_loss_ratio"],
        semantic_distance=sem if sem is not None else float("nan"),
    )

    _report_status(status, "Assessing EEDI risk...")
    result = asyncio.run(call_judge(DEFAULT_JUDGE_MODEL, system_prompt, user_prompt))
    if result is None:
        _report_status(status, "Judge call failed after retries.")
        return None, None, "LLM judge call failed after retries."

    _report_status(status, "Finalising the assessment...")
    return result.overall_eedi_risk_score, result.flagged_dimension, result.justification


def _clear_inputs() -> None:
    """Clear both text inputs without changing the rest of the app configuration."""
    st.session_state["response_text"] = ""
    st.session_state["reference_text"] = ""


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bias Busters", page_icon="🐾", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    :root {
        --brand-yellow: #FFD200;
        --brand-yellow-soft: #FFF7CC;
        --ink: #202124;
        --muted-text: #5F6368;
        --page-bg: #F6F7F4;
        --surface: #FFFFFF;
        --border: #E2E4DE;
        --info: #4477AA;
        --low-risk: #228833;
        --moderate-risk: #CCBB44;
        --high-risk: #EE6677;
        --severe-risk: #AA3377;
    }

    .stApp { background: var(--page-bg); color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem; }

    .brand-header {
        background: var(--brand-yellow);
        border: 1px solid #D5AF00;
        border-radius: 14px;
        padding: 0.75rem 1.35rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    .brand-header img { height: 78px; width: auto; object-fit: contain; }
    .brand-header h1 { margin: 0; font-size: 2rem; line-height: 1.1; font-weight: 850; color: var(--ink); }
    .brand-header p { margin: 0.3rem 0 0; color: #353535; font-size: 1rem; }

    .intro-copy { color: var(--muted-text); font-size: 1rem; margin-bottom: 1rem; }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 4px solid var(--info);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        min-height: 126px;
    }
    .metric-card .mc-name {
        color: var(--info);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .metric-card .mc-desc { color: var(--muted-text); font-size: 0.9rem; margin-top: 0.45rem; }

    .section-label {
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }

    div[data-testid="stTextArea"] textarea {
        background: var(--surface);
        border-color: var(--border);
        border-radius: 10px;
    }

    .risk-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 8px solid #777;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 0.8rem 0 1rem;
        box-shadow: 0 2px 8px rgba(25, 30, 25, 0.05);
    }
    .risk-1 { border-left-color: var(--low-risk); background: #F2FAF4; }
    .risk-2 { border-left-color: #66AA55; background: #F6FBF4; }
    .risk-3 { border-left-color: var(--moderate-risk); background: #FFFBEA; }
    .risk-4 { border-left-color: var(--high-risk); background: #FFF3F4; }
    .risk-5 { border-left-color: var(--severe-risk); background: #FBF1F8; }
    .risk-unknown { border-left-color: #777; }
    .risk-eyebrow {
        color: var(--muted-text);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .risk-score { font-size: 2.8rem; line-height: 1; font-weight: 850; margin: 0.35rem 0 0; }
    .risk-label { font-size: 1.25rem; font-weight: 750; margin-bottom: 0.8rem; }
    .risk-detail { color: #353535; margin: 0.25rem 0; }

    .screening-note {
        background: var(--brand-yellow-soft);
        border: 1px solid #EADFA8;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #4A442A;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px !important;
        font-weight: 750 !important;
        min-height: 2.8rem;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: var(--brand-yellow) !important;
        color: var(--ink) !important;
        border: 1px solid #D5AF00 !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: #EABD00 !important;
        border-color: #B99600 !important;
    }

    @media (max-width: 720px) {
        .brand-header { align-items: flex-start; gap: 0.8rem; }
        .brand-header img { height: 58px; }
        .brand-header h1 { font-size: 1.55rem; }
        .block-container { padding: 1rem; }
    }
</style>

""",
    unsafe_allow_html=True,
)

_logo_b64 = base64.b64encode((_APP_DIR / "image" / "DogsTrust_Tag_Logo_RGB.png").read_bytes()).decode()

st.markdown(
    f"""
<div class="brand-header">
    <img src="data:image/png;base64,{_logo_b64}" alt="Dogs Trust logo" />
    <div>
        <h1>Bias Busters</h1>
        <p>EEDI screening for AI-generated fundraising letter · Dogs Trust</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="intro-copy">Review an AI-generated fundraising letter for recurring linguistic patterns, '
    "beneficiary choices, and potential EEDI concerns. Results are screening evidence and require human judgement.</p>",
    unsafe_allow_html=True,
)

# ── Input section ─────────────────────────────────────────────────────────────

st.subheader("Analyse a fundraising letter")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        '<div class="section-label">AI-generated letter <span style="color:#AA3377">*</span></div>',
        unsafe_allow_html=True,
    )
    response_text = st.text_area(
        label="AI-Generated Output",
        key="response_text",
        label_visibility="collapsed",
        placeholder="Paste the AI-generated fundraising letter here…",
        height=280,
    )

with col_right:
    st.markdown(
        '<div class="section-label">Reference letter '
        '<span style="color:#888; font-weight:400">(optional)</span></div>',
        unsafe_allow_html=True,
    )
    reference_text = st.text_area(
        label="Reference Text",
        key="reference_text",
        label_visibility="collapsed",
        placeholder="Optional: paste the reference-condition letter for semantic comparison…",
        height=240,
    )
    st.caption(
        "A reference letter enables semantic-distance scoring. Leave this blank to analyse the other three measures."
    )

st.markdown("")
run_col, clear_col = st.columns([4, 1])
with run_col:
    run = st.button(
        "Analyse EEDI risk",
        type="primary",
        disabled=not response_text.strip(),
        use_container_width=True,
    )
with clear_col:
    st.button("Clear", use_container_width=True, on_click=_clear_inputs)

with st.expander("How the assessment works"):
    st.markdown(
        """
        1. Paste an AI-generated fundraising letter.
        2. Optionally add a reference-condition letter to calculate semantic distance.
        3. Run the assessment and review the EEDI screening result, linguistic measures, and dog details.

        The screening combines **agency**, **formality**, **gain/loss framing**, and optional
        **semantic distance** with an LLM-based assessment. It does not establish that a letter
        is discriminatory and should not replace professional review.

        | EEDI dimension | Working definition |
        |---|---|
        | **Equality** | Consistent participation without unjustified disadvantage |
        | **Equity** | Appropriate adaptation where circumstances or barriers justify it |
        | **Diversity** | Recognition of different identities without stereotyping |
        | **Inclusion** | Meaningful participation without exclusion or condescension |
        """
    )

# ── Results ───────────────────────────────────────────────────────────────────

if run and response_text.strip():
    clean_response = response_text.strip()
    ref = reference_text.strip() or None

    with st.status("Analysing language and dog details…", expanded=False) as status:
        scores = _score_text(clean_response, ref, status=status)
        dog_details = _extract_dog_details(clean_response, status=status)
        if dog_details is None:
            status.write("Dog-detail extraction failed after retries.")
            status.update(label="Language metrics complete; dog extraction unavailable", state="complete")
        else:
            status.update(label="Language and dog analysis complete", state="complete")

    with st.status("Assessing EEDI risk…", expanded=False) as status:
        eedi_score, flagged_dim, justification = _run_judge(clean_response, scores, status=status)
        status.update(label="EEDI judge complete", state="complete")

    st.divider()
    st.subheader("Assessment result")

    label, _ = _eedi_badge(eedi_score)
    risk_class = f"risk-{eedi_score}" if eedi_score is not None else "risk-unknown"
    score_display = str(eedi_score) if eedi_score is not None else "—"
    safe_label = html.escape(label)
    safe_dimension = html.escape(flagged_dim or "None identified")
    safe_justification = html.escape(justification or "No justification was returned.")

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <div class="risk-eyebrow">EEDI screening result</div>
            <div class="risk-score">
                {score_display}<span style="font-size:1rem;font-weight:600;color:#666;"> / 5</span>
            </div>
            <div class="risk-label">{safe_label}</div>
            <div class="risk-detail"><strong>Flagged area:</strong> {safe_dimension}</div>
            <div class="risk-detail"><strong>Assessment:</strong> {safe_justification}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if eedi_score and eedi_score >= 3:
        st.warning(
            "**Recommended action:** Review the flagged area before publishing. "
            "For scores 4–5, revise the letter and consider regenerating it with a more neutral prompt."
        )
    else:
        st.markdown(
            '<div class="screening-note"><strong>Screening reminder:</strong> A low score does not prove that the '
            "letter is free from EEDI concerns. Retain human review before publication.</div>",
            unsafe_allow_html=True,
        )

    metrics_tab, dog_tab, technical_tab = st.tabs(["Linguistic measures", "Dog details", "Technical details"])

    with metrics_tab:
        st.caption("These measures describe selected features of the letter; none is a standalone measure of bias.")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Agency ratio", f"{scores['agency_ratio']:+.3f}")
            st.progress(max(0.0, min(1.0, (scores["agency_ratio"] + 1) / 2)))
            st.caption("−1 lower-agency forms · +1 higher-agency forms")
        with m2:
            st.metric("Formality score", f"{scores['formality_score']:.1f}")
            st.progress(max(0.0, min(1.0, scores["formality_score"] / 100)))
            st.caption("0 informal · 100 formal")
        with m3:
            st.metric("Gain/loss ratio", f"{scores['gain_loss_ratio']:+.3f}")
            st.progress(max(0.0, min(1.0, (scores["gain_loss_ratio"] + 1) / 2)))
            st.caption("−1 loss-framed · +1 gain-framed")
        with m4:
            if scores["semantic_distance"] is None:
                st.metric("Semantic distance", "—")
                st.progress(0.0)
                st.caption("Add a reference letter to calculate this measure")
            else:
                st.metric("Semantic distance", f"{scores['semantic_distance']:.3f}")
                st.progress(max(0.0, min(1.0, scores["semantic_distance"])))
                st.caption("0 identical · 1 very different")

        st.markdown("#### What the measures mean")
        explanation_left, explanation_right = st.columns(2)
        with explanation_left:
            st.markdown(
                """
                **Agency ratio**
                Compares the number of higher-agency and lower-agency verb forms in the
                complete letter. Positive scores indicate relatively more higher-agency
                forms, while negative scores indicate relatively more lower-agency forms.
                The score does not identify who the verb describes, and neither end of the
                scale is inherently biased.

                **Formality score**
                Describes the letter's linguistic register. Lower scores indicate more
                conversational and context-dependent language; higher scores indicate more
                formal and informationally explicit language. Formality is not inherently
                good or bad, but systematic differences may suggest that groups are being
                addressed differently.
                """
            )
        with explanation_right:
            st.markdown(
                """
                **Gain/loss ratio**
                Compares sentences focused on desirable outcomes, such as recovery or
                rehoming, with sentences focused on suffering, danger, or the consequences
                of inaction. Positive scores indicate relatively more gain framing and
                negative scores indicate relatively more loss framing. A score of zero may
                also mean that no classifiable framing was found.

                **Semantic distance**
                Measures how far the overall meaning of the letter differs from the supplied
                reference letter. Zero indicates identical semantic content, while a higher
                score indicates greater difference. It shows that content changed, but not
                what changed or whether the difference is harmful.
                """
            )

    with dog_tab:
        st.caption("Dog characteristics are extracted separately and are not supplied to the EEDI judge.")
        if dog_details is None:
            st.warning("Dog details could not be extracted; the other assessment results remain available.")
        else:
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Dog name**  \n{dog_details.dog_name}")
                st.markdown(f"**Breed stated**  \n{dog_details.breed_mentioned}")
                st.markdown(f"**KC breed group**  \n{dog_details.kc_breed_group}")
            with d2:
                st.markdown(f"**Size**  \n{dog_details.size_category}")
                st.markdown(f"**Life stage**  \n{dog_details.life_stage}")

    with technical_tab:
        st.markdown(f"**Judge model:** `{DEFAULT_JUDGE_MODEL}`")
        st.markdown(f"**Dog-extraction model:** `{DEFAULT_BREED_MODEL}`")
        st.markdown(f"**Reference letter supplied:** {'Yes' if ref else 'No'}")
        with st.expander("Raw linguistic values"):
            st.json(scores)
