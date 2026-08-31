"""Provide shared asynchronous LLM calls with retry handling."""

import logging

import litellm
import litellm.exceptions
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from constants import RETRY_ATTEMPTS

logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type((litellm.exceptions.RateLimitError, litellm.exceptions.APIError)),
    wait=wait_exponential(multiplier=2, min=8, max=120),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _acall(model: str, messages: list[dict], **completion_kwargs) -> str:
    """Make a single litellm call, raising on failure for tenacity to handle.

    This must raise (not catch) on failure — @retry only retries when the
    wrapped call raises a matching exception. Catching here would make the
    call always look "successful" to tenacity and the retry would never fire.

    Args:
        model: litellm model string (e.g. "openai/gpt-4o").
        messages: Chat messages to send.
        **completion_kwargs: Extra keyword args forwarded to litellm.acompletion
            (e.g. temperature, max_tokens).

    Returns:
        The model's response text.
    """
    response = await litellm.acompletion(model=model, messages=messages, **completion_kwargs)
    return response.choices[0].message.content.strip()


async def acall_with_retry(model: str, messages: list[dict], **completion_kwargs) -> str | None:
    """Call an LLM with tenacity-managed exponential back-off retry.

    Args:
        model: litellm model string (e.g. "openai/gpt-4o").
        messages: Chat messages to send.
        **completion_kwargs: Extra keyword args forwarded to litellm.acompletion
            (e.g. temperature, max_tokens).

    Returns:
        The model's response text, or None if all attempts fail or an auth error occurs.
    """
    try:
        return await _acall(model, messages, **completion_kwargs)
    except litellm.exceptions.AuthenticationError as exc:
        logger.error("Auth error for %s: %s", model, exc)
        return None
    except Exception as exc:
        logger.error("All retries exhausted for %s: %s", model, exc)
        return None
