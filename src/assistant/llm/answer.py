"""Select user-visible content from completed model answers."""

from assistant.llm.base import LLMError


def answer_text(content: str) -> str:
    """Remove leading think blocks; reject reasoning without a final answer."""
    answer = content
    while answer.lstrip().startswith("<think>"):
        candidate = answer.lstrip()
        end = candidate.find("</think>", len("<think>"))
        if end == -1:
            raise LLMError("The model returned reasoning without a complete answer.")
        answer = candidate[end + len("</think>") :].lstrip()
    if not answer.strip():
        raise LLMError("The model returned no answer text.")
    return answer
