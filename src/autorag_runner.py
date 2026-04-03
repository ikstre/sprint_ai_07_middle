"""
Runtime wrapper for AutoRAG optimized trial.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoRAGRuntime:
    trial_dir: str

    def __post_init__(self) -> None:
        try:
            from autorag.deploy import Runner
        except ImportError as exc:
            raise ImportError(
                "AutoRAG is not installed. Install with `pip install AutoRAG`."
            ) from exc

        self._runner = Runner.from_trial_folder(self.trial_dir)

    def ask(self, question: str):
        return self._runner.run(question)

