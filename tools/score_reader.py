"""Score `read_paper` against a hand-labelled truth file.

Three numbers, always reported separately, because they fail differently:

  found        the labelled question was opened at all, with the right number
               on the right start page
  spurious     boundaries the reader invented, which is the failure that
               silently splits one question into two
  exact pages  the whole page set matched, which is the cross-page-break test

Do not collapse these into one accuracy figure. A single number hides the
failure that matters: a reader that finds every question but gets the page sets
wrong looks excellent and cuts the corpus wrongly on every page break.

    python3 tools/score_reader.py tests/fixtures/public_papers_truth.json
    python3 tools/score_reader.py my-yardstick.json --papers ~/papers
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from reader import read_paper_with_report                       # noqa: E402


def score(predicted, expected):
    """(found, spurious, exact_pages) for one paper."""
    by_number = {question["number"]: question for question in predicted}
    labelled = {want["n"] for want in expected}
    found = exact = 0
    for want in expected:
        got = by_number.get(want["n"])
        if got and got["start_page"] == want["pages"][0]:
            found += 1
            if list(got["pages"]) == list(want["pages"]):
                exact += 1
    spurious = len([q for q in predicted if q["number"] not in labelled])
    return found, spurious, exact


def resolve(spec, papers_dir):
    """Truth files name papers relative to the papers directory, so the PDFs
    themselves stay out of the repository. A relative path may carry
    subdirectories - the scan proxies live in one - but it may not escape."""
    named = pathlib.PurePosixPath(spec["pdf"])
    if named.is_absolute() or ".." in named.parts:
        raise ValueError(f"a truth file may not name {spec['pdf']}: paths are "
                         "relative to the papers directory")
    return pathlib.Path(papers_dir).joinpath(*named.parts)


def score_truth(truth, papers_dir, rung=None):
    """Score every case in a truth file. Yields one result dict per case."""
    for case, spec in truth.items():
        if not spec.get("questions"):
            continue
        path = resolve(spec, papers_dir)
        if not path.exists():
            yield {"case": case, "missing": str(path)}
            continue
        # strict off: scoring a reader that refuses to emit tells us nothing
        predicted, report = read_paper_with_report(
            path, spec.get("pages"), rung=rung, strict=False)
        found, spurious, exact = score(predicted, spec["questions"])
        yield {"case": case, "total": len(spec["questions"]), "found": found,
               "spurious": spurious, "exact": exact, "rung": report["rung"],
               "column": report["question_column"],
               "trustworthy": report["trustworthy"],
               "problems": report["problems"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth")
    parser.add_argument("--papers", default="tests/fixtures/papers",
                        help="directory holding the PDFs the truth file names")
    parser.add_argument("--rung", type=int, choices=(1, 2),
                        help="force a rung instead of choosing automatically")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    truth = json.loads(pathlib.Path(arguments.truth).read_text())
    results = list(score_truth(truth, arguments.papers, arguments.rung))

    if arguments.json:
        print(json.dumps(results, indent=1))
        return

    print(f"{'case':<20} {'rung':>4} {'questions':>9} {'found':>8} "
          f"{'spurious':>8} {'exact pages':>12}")
    totals = {"total": 0, "found": 0, "spurious": 0, "exact": 0}
    for result in results:
        if "missing" in result:
            print(f"{result['case']:<20} paper not found: {result['missing']}")
            continue
        print(f"{result['case']:<20} {result['rung']:>4} {result['total']:>9} "
              f"{result['found']:>5}/{result['total']:<2} {result['spurious']:>8} "
              f"{result['exact']:>7}/{result['total']:<4}")
        for key in totals:
            totals[key] += result[key]
    if totals["total"]:
        print(f"{'TOTAL':<20} {'':>4} {totals['total']:>9} "
              f"{totals['found']:>5}/{totals['total']:<2} {totals['spurious']:>8} "
              f"{totals['exact']:>7}/{totals['total']:<4}")


if __name__ == "__main__":
    main()
