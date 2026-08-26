"""Count how much of each question the transformer's tokeniser actually sees.

src/train_transformer.py truncates at MAX_TOKENS, while Approach 1 reads every
word of every question. If a real share of the corpus runs past the limit then
the two approaches are not being asked the same question, and the comparison
between them is worth less than it looks.

So this counts rather than assumes: the length distribution over the whole
corpus, how many questions the current limit cuts, and how much of them it
cuts. The answer belongs in a measurement that can be re-run when the corpus
grows, not in a comment asserting that questions are short.

Run from the project root:  python3 tools/measure_token_lengths.py
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import normalise_question  # noqa: E402
from paths import PRIVATE_DATA  # noqa: E402
from train_transformer import BASE_MODEL, MAX_TOKENS  # noqa: E402

# The limits worth pricing: the one in use, the next power of two, and the
# model's own ceiling, which is also 512 for DistilBERT.
LIMITS_TO_PRICE = (128, 256, 512)

PERCENTILES = (50, 75, 90, 95, 99, 100)


def read_question_lengths():
    """Token counts for every question in the corpus, longest first.

    Counted without truncation and without padding, so the number is the
    question's real length in the tokeniser's terms, including the two special
    tokens the model always adds.
    """
    question_tokeniser = AutoTokenizer.from_pretrained(BASE_MODEL)
    lengths = []
    with (PRIVATE_DATA / "questions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = normalise_question(json.loads(line)["text"])
            lengths.append(len(question_tokeniser(text)["input_ids"]))
    return sorted(lengths, reverse=True)


def percentile(sorted_descending, share):
    """The length at a given percentile, from a descending list."""
    ascending = sorted_descending[::-1]
    index = min(int(round(share / 100 * len(ascending))), len(ascending)) - 1
    return ascending[max(index, 0)]


def main():
    lengths = read_question_lengths()
    total = len(lengths)

    print(f"{'=' * 72}")
    print(f"TOKEN LENGTHS, {total} questions through {BASE_MODEL}")
    print(f"{'=' * 72}")
    print(f"  shortest {min(lengths)}   mean {sum(lengths) / total:.0f}   "
          f"longest {max(lengths)}")
    for share in PERCENTILES:
        print(f"  {share:>3}th percentile   {percentile(lengths, share):>4} tokens")

    print(f"\n{'-' * 72}\nWHAT EACH LIMIT WOULD CUT\n{'-' * 72}")
    print(f"  {'limit':>6}{'questions cut':>16}{'share':>9}"
          f"{'tokens lost':>14}{'of all tokens':>16}")
    all_tokens = sum(lengths)
    for limit in LIMITS_TO_PRICE:
        over = [length for length in lengths if length > limit]
        lost = sum(length - limit for length in over)
        print(f"  {limit:>6}{len(over):>16}{len(over) / total:>9.1%}"
              f"{lost:>14}{lost / all_tokens:>16.1%}")

    over_current = [length for length in lengths if length > MAX_TOKENS]
    print(f"\n  at the limit in use ({MAX_TOKENS}): {len(over_current)} questions "
          f"are cut")
    if over_current:
        worst = max(over_current)
        print(f"  the longest loses {worst - MAX_TOKENS} tokens of its {worst}")


if __name__ == "__main__":
    main()
