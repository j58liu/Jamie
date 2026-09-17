"""Run the model evaluation from a checkout: python evals/runner.py --runs 5."""

import sys

from assistant.cli import app

if __name__ == "__main__":
    app(args=["eval", "model", *sys.argv[1:]], prog_name="jamie")
