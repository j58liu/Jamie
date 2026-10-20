"""Select user-visible content from completed model answers."""

from assistant.llm.base import LLMError


class AnswerStream:
    """Hide leading reasoning blocks, even when their tags cross chunk boundaries."""

    def __init__(self) -> None:
        self._pending = ""
        self._reasoning = False
        self._answer = False

    def feed(self, text: str) -> str:
        if self._answer:
            return text
        self._pending += text
        while True:
            if self._reasoning:
                end = self._pending.find("</think>")
                if end < 0:
                    self._pending = self._pending[-len("</think>") :]
                    return ""
                self._pending = self._pending[end + len("</think>") :].lstrip()
                self._reasoning = False
            candidate = self._pending.lstrip()
            if not candidate or "<think>".startswith(candidate):
                return ""
            if candidate.startswith("<think>"):
                self._pending = candidate[len("<think>") :]
                self._reasoning = True
                continue
            self._answer = True
            result, self._pending = self._pending, ""
            return result

    def finish(self) -> str:
        if self._reasoning or self._pending.lstrip().startswith("<think"):
            raise LLMError("The model returned reasoning without a complete answer.")
        result, self._pending = self._pending, ""
        return result


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
