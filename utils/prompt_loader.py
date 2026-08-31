"""Load and save prompt templates."""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent.parent / "prompt"


def load_prompt(name: str, subdir: str = "experiments") -> str:
    """Load a prompt file by stem from the given subdirectory.

    Args:
        name: Filename stem (without .txt extension).
        subdir: Subdirectory under prompt/. Defaults to "experiments".

    Returns:
        The prompt text, stripped of leading/trailing whitespace.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPT_DIR / subdir / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def save_prompt(name: str, text: str, subdir: str = "built") -> Path:
    """Write a fully rendered prompt to prompt/<subdir>/<name>.txt.

    Args:
        name: Filename stem (without .txt extension).
        text: Prompt text to write.
        subdir: Subdirectory under prompt/. Defaults to "built".

    Returns:
        The path the prompt was written to.
    """
    path = _PROMPT_DIR / subdir / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
