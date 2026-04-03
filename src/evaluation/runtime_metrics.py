"""
Runtime/token metric helpers.
"""


def usage_to_tokens(result: dict) -> dict:
    usage = result.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if total_tokens is None:
        estimated_total = int((result.get("context_char_len", 0) + result.get("answer_char_len", 0)) / 4)
        total_tokens = estimated_total

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

