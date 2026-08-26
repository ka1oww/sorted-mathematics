"""Extract labelled rows from the LaTeX sources (1 and 2 in DATA-PLAN.md).

Source 1 is the revision packages: one question per `*_q.tex` under
`revision-packages/<CHAPTER>/parts/`, each carrying a `\\qend{...}` provenance
tag and per-part `\\mk{n}` mark counts. Source 2 is the tutorials: one `.tex`
paper per chapter, questions delimited by `\\Q`, where `\\qend{n}` carries the
question's total marks rather than provenance.

It never compiles LaTeX; compiling would drop .aux and .log files beside the
sources, which are read-only.

Output: data/private/ember_latex.jsonl, one JSON object per question.

Run: python3 src/extract_latex.py
"""

import collections
import json
import re

from chapters import CHAPTER_SLUGS, resolve
from paths import PRIVATE_DATA, assert_inside_project, ember_source_root

SOURCE_ROOT = ember_source_root()
OUTPUT_PATH = PRIVATE_DATA / "ember_latex.jsonl"

# Tutorial folders that hold only PDFs. Their questions are extracted by the
# PDF pipeline (extract_pdf.py), so this script must not touch them. Listing
# them explicitly means an unexpected new folder still gets flagged below.
PDF_ONLY_TUTORIALS = {
    "binomial-distribution",
    "discrete-random-variables",
    "hypothesis-testing",
    "normal-distribution",
    "PNC",
    "probability",
    "sampling",
}

# ---------------------------------------------------------------------------
# Normalisation: LaTeX -> plain prose
# ---------------------------------------------------------------------------

# Commands whose meaning survives as an English word or plain symbol. The
# encoder this feeds has effectively never seen LaTeX, so `\int` is noise
# while `integral` carries a rich pre-trained representation. This table was
# built from a frequency count over the actual corpus, not from imagination;
# anything not handled here is dropped and reported by report_dropped_commands.
WORD_MAP = {
    "int": "integral",
    "sum": "sum",
    "lim": "limit",
    "infty": "infinity",
    # Trig and log names read the same with the backslash removed.
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sec": "sec",
    "cot": "cot",
    "ln": "ln",
    "arg": "arg",
    # Greek letters: teachers write these out as words.
    "pi": "pi",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "sigma": "sigma",
    "phi": "phi",
    "omega": "omega",
    "Pi": "Pi",
    # Relations and operators.
    "le": "<=",
    "leq": "<=",
    "ge": ">=",
    "geq": ">=",
    "geqslant": ">=",
    "neq": "!=",
    "ne": "!=",
    "in": "in",
    "cap": "intersection",
    "cup": "union",
    "setminus": "excluding",
    "to": "->",
    "mapsto": "->",
    "pm": "+/-",
    "times": "times",
    # In this corpus \cdot is almost always the vector dot product.
    "cdot": "dot",
    "approx": "approximately",
    "propto": "proportional to",
    "mid": "|",
    "lvert": "|",
    "rvert": "|",
    "angle": "angle",
    "triangle": "triangle",
    "ldots": "...",
    "cdots": "...",
    "vdots": "...",
    "ast": "*",
    "circ": "o",  # survives only as f \circ g composition; degrees handled earlier
    "R": "R",  # \R is \mathbb{R} in this corpus's preambles
    "textemdash": "-",
    "textbar": "|",
    "crossp": "cross",
    # Single-letter bold-vector shorthands defined in the tutorial preambles
    # (\newcommand{\va}{\mathbf{a}} and so on). Bold is presentation; the
    # letter is the content.
    "va": "a", "vb": "b", "vc": "c", "vd": "d", "vi": "i", "vj": "j",
    "vk": "k", "vn": "n", "vp": "p", "vq": "q", "vr": "r", "vu": "u",
    "vv": "v",
}

# Wrappers where only the argument matters: formatting, fonts, colour.
KEEP_CONTENT_WRAPPERS = (
    "text", "textbf", "textit", "emph", "mathrm", "mathbf", "mathbb",
    "mbox", "underline", "uline", "operatorname", "bar", "hat",
)

# Commands that are pure layout and vanish along with their argument.
DROP_WITH_ARG = (
    "mk", "qend", "embersection", "includegraphics", "vspace", "hspace",
    "setcounter", "color", "fontsize", "addfontfeature", "thispagestyle",
    "rule", "pagestyle", "label", "ref",
)

# Bare commands that are pure layout: spacing, sizing, delimiter sizing.
DROP_BARE = (
    "Q", "item", "left", "right", "big", "bigl", "bigr", "Big", "Bigl",
    "Bigr", "displaystyle", "quad", "qquad", "noindent", "par", "hfill",
    "medskip", "smallskip", "small", "footnotesize", "scriptsize",
    "centering", "selectfont", "bfseries", "itshape", "georgia",
    "arraybackslash", "hline", "cline", "arraystretch", "ignorespaces",
    "dots",
)

# Dropped commands are counted here so the long tail can be reported instead
# of disappearing silently.
dropped_commands = collections.Counter()


def strip_comments(latex):
    """Remove % comments (but keep escaped \\% percent signs)."""
    return re.sub(r"(?<!\\)%.*", "", latex)


def read_brace_group(text, pos):
    """Read one balanced `{...}` group starting at `pos`; return (content, end)."""
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return text[pos + 1:], len(text)


def rewrite_command(text, name, nargs, build, optional=False):
    """Replace every `\\name{...}...` with build(args).

    Regexes cannot count braces, so arguments are read with a balanced-brace
    scan; a \\frac nested inside another \\frac's argument is rewritten on a
    later pass of the loop. `optional` reads a leading `[...]`
    argument (as in \\sqrt[3]{x}) into args[0], or None when absent.
    """
    marker = re.compile(r"\\" + name + r"(?![a-zA-Z])\s*")
    while True:
        match = marker.search(text)
        if match is None:
            return text
        pos = match.end()
        args = []
        if optional:
            if pos < len(text) and text[pos] == "[":
                close = text.find("]", pos)
                args.append(text[pos + 1:close])
                pos = close + 1
            else:
                args.append(None)
        complete = True
        for _ in range(nargs):
            if pos < len(text) and text[pos] == "{":
                content, pos = read_brace_group(text, pos)
                args.append(content)
            else:
                complete = False
                break
        if complete:
            text = text[:match.start()] + build(args) + text[pos:]
        else:
            # Malformed usage: drop the command token alone so the scan
            # cannot loop forever on it.
            text = text[:match.start()] + " " + text[match.end():]


def normalise_to_prose(latex):
    """Turn a LaTeX question into the plain prose a teacher would paste.

    The order matters: comments first (so commented-out LaTeX never leaks),
    then whole figure environments, then macros from most specific to most
    generic, and finally leftover syntax characters.
    """
    text = strip_comments(latex)

    # Figures carry no usable text: diagrams live in tikzpicture environments
    # and their coordinates/styling would only add noise.
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", text, flags=re.DOTALL)

    # Environment shells. tabular/minipage take arguments; the generic form
    # also eats option lists like [label=(\alph*), itemsep=8pt].
    text = re.sub(r"\\begin\{(?:tabular|minipage)\}(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\item\b(?:\[[^\]]*\])?", " ", text)

    # Degrees before the word map, because a bare \circ later means function
    # composition instead.
    text = re.sub(r"\^\{?\\circ\}?", " degrees", text)

    # Fractions and roots become spoken forms. Every replacement starts with
    # a space so a preceding command name can never fuse with the result.
    for frac in ("frac", "dfrac", "tfrac"):
        text = rewrite_command(text, frac, 2, lambda a: f" {a[0]} / {a[1]}")
    text = rewrite_command(text, "sqrt", 1, optional=True, build=lambda a:
                           f" square root of {a[1]}" if a[0] is None
                           else f" root {a[0]} of {a[1]}")

    # Corpus-specific macros from the Ember preambles.
    text = rewrite_command(text, "colvec", 3, lambda a: f" ({a[0]}, {a[1]}, {a[2]})")
    text = rewrite_command(text, "OAvec", 1, lambda a: f" vector {a[0]}")
    text = rewrite_command(text, "oa", 1, lambda a: f" vector {a[0]}")
    text = rewrite_command(text, "uv", 1, lambda a: f" unit vector {a[0]}")
    text = rewrite_command(text, "md", 1, lambda a: f" |{a[0]}|")
    text = rewrite_command(text, "textcolor", 2, lambda a: f" {a[1]}")
    text = rewrite_command(text, "renewcommand", 2, lambda a: " ")

    for name in DROP_WITH_ARG:
        text = re.sub(r"\\" + name + r"\b(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)

    for name in KEEP_CONTENT_WRAPPERS:
        text = rewrite_command(text, name, 1, lambda a: f" {a[0]}")

    def replace_bare(match):
        name = match.group(1)
        if name in WORD_MAP:
            return " " + WORD_MAP[name] + " "
        if name in DROP_BARE:
            return " "
        dropped_commands[name] += 1
        return " "

    # \\ is a line break, then every remaining \command is either mapped,
    # deliberately dropped, or counted as unhandled long tail.
    text = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\([a-zA-Z]+)", replace_bare, text)

    # Display-math delimiters, spacing commands and accent marks.
    text = re.sub(r"\\[\[\]]", " ", text)
    text = re.sub(r"\\[,;!: ]", " ", text)
    text = text.replace("\\'", "")
    # Escaped characters are literal content: a money amount's \$, a \%, a
    # set's \{...\}. They must survive the strip of math-mode $ and
    # grouping braces below. Placeholders carry them through that strip.
    text = text.replace("\\$", "\x00").replace("\\{", "\x01").replace("\\}", "\x02")
    text = re.sub(r"\\([%&_#])", r"\1", text)
    text = re.sub(r"[${}~]|&", " ", text)
    text = text.replace("\x00", "$").replace("\x01", "{").replace("\x02", "}")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Provenance parsing (revision packages only)
# ---------------------------------------------------------------------------

EXAM_NAMES = {
    "prelim", "prelims", "mye", "mya", "bt", "bt1", "bt2", "ct", "ct1",
    "ct2", "mid year ct", "revision package", "promo", "'a' lvls",
}
unparsed_provenance_tokens = collections.Counter()


def parse_provenance(tag):
    """Parse a `\\qend{2024 / ABCJC / JC2 / Prelim / P2 / Q1}` tag.

    The field count varies (some tags omit level or exam, two swap school and
    exam, a few end in a topic name instead of a question number), so each
    token is classified by what it looks like rather than by its position.
    Anything unclassifiable is counted and reported, never guessed.
    """
    fields = {"year": None, "school": None, "level": None,
              "exam": None, "paper": None, "question_no": None}

    modified = bool(re.search(r"\(\s*modified\s*\)\s*$", tag))
    tag = re.sub(r"\(\s*modified\s*\)\s*$", "", tag).strip()

    # A few tags are a single space-separated phrase rather than slash-separated
    # fields. Matched by shape so no school is named here.
    phrase = re.fullmatch(r"([A-Z]{2,6}(?:\(JC\))?)\s+((?:JC|SH|PU|J)[123])\s+(Revision Package)",
                          tag)
    if phrase:
        fields.update(school=phrase.group(1), level=phrase.group(2),
                      exam=phrase.group(3))
        return fields, modified

    for token in [t.strip() for t in tag.split(" / ") if t.strip()]:
        lower = token.lower()
        if fields["year"] is None and re.fullmatch(r"(19|20)\d{2}", token):
            fields["year"] = int(token)
        elif fields["level"] is None and re.fullmatch(r"(?:JC|SH|PU|J)[123]", token):
            fields["level"] = token
        elif fields["exam"] is None and lower in EXAM_NAMES:
            fields["exam"] = token
        elif fields["paper"] is None and re.fullmatch(r"P\d|I|II", token):
            fields["paper"] = token
        # A bare number is a question number only when it is small; a long
        # number is something else (e.g. the syllabus code 9740).
        elif fields["question_no"] is None and re.fullmatch(
                r"(?:Q\d+|\d{1,2})\s*(\([a-z]+\)\s*)*", token):
            fields["question_no"] = token
        elif fields["school"] is None and re.fullmatch(r"[A-Z]{2,6}(\(JC\))?", token):
            fields["school"] = token
        else:
            unparsed_provenance_tokens[token] += 1
    return fields, modified


# ---------------------------------------------------------------------------
# Source 1: revision packages, one question per *_q.tex
# ---------------------------------------------------------------------------

def extract_revision_packages():
    rows = []
    root = SOURCE_ROOT / "revision-packages"
    for chapter_dir in sorted(root.iterdir()):
        if not chapter_dir.is_dir():
            continue
        chapter = resolve(chapter_dir.name)
        if chapter is None:
            # Every directory here should be a chapter; an unknown name means
            # chapters.py is missing an alias, which needs a human decision.
            raise SystemExit(f"unrecognised chapter folder: {chapter_dir}")
        for tex_path in sorted(chapter_dir.glob("parts/*_q.tex")):
            raw = tex_path.read_text(encoding="utf-8")

            qend = re.findall(r"\\qend\{([^}]*)\}", raw)
            if len(qend) != 1:
                raise SystemExit(f"expected exactly one \\qend in {tex_path}, found {len(qend)}")
            provenance, modified = parse_provenance(qend[0])

            marks = [int(m) for m in re.findall(r"\\mk\{(\d+)\}", raw)]

            rows.append({
                "id": f"rp:{chapter_dir.name}:{tex_path.stem.removesuffix('_q')}",
                "chapter": chapter,
                "text": normalise_to_prose(raw),
                "raw_text": raw,
                "source": "revision_package",
                **provenance,
                "modified": modified,
                "marks": sum(marks) if marks else None,
            })
    return rows


# ---------------------------------------------------------------------------
# Source 2: tutorials, several questions per .tex split on \Q
# ---------------------------------------------------------------------------

def split_tutorial_questions(body):
    """Split a tutorial body into raw question chunks.

    A question runs from its `\\Q` line to its `\\qend{...}` line inclusive.
    The comment-stripped view decides where questions start and end (so a
    commented-out `\\Q` cannot create a phantom question), but the returned
    chunks are the original lines, exactly as found.
    """
    questions = []
    current = None
    for line in body.splitlines():
        meaning = strip_comments(line).strip()
        if re.match(r"\\Q\b", meaning):
            if current is not None:
                questions.append("\n".join(current))
            current = [line]
        elif current is not None:
            # A section heading between questions means the previous question
            # had no \qend line; close it rather than absorbing the heading.
            if meaning.startswith("\\embersection") or meaning.startswith("\\end{document}"):
                questions.append("\n".join(current))
                current = None
            else:
                current.append(line)
                if "\\qend{" in meaning:
                    questions.append("\n".join(current))
                    current = None
    if current is not None:
        questions.append("\n".join(current))
    return questions


def extract_tutorials():
    rows = []
    root = SOURCE_ROOT / "tutorials"
    for chapter_dir in sorted(root.iterdir()):
        if not chapter_dir.is_dir():
            continue
        tex_files = [p for p in sorted(chapter_dir.glob("*.tex"))
                     if "solutions" not in p.name.lower()]
        if not tex_files:
            if chapter_dir.name not in PDF_ONLY_TUTORIALS:
                raise SystemExit(f"tutorial folder has no question .tex and is "
                                 f"not a known PDF-only chapter: {chapter_dir}")
            continue
        chapter = resolve(chapter_dir.name)
        if chapter is None:
            raise SystemExit(f"unrecognised chapter folder: {chapter_dir}")
        for tex_path in tex_files:
            full = tex_path.read_text(encoding="utf-8")
            # The preamble defines \Q via \newcommand; only the document body
            # uses it as a question marker.
            body = full.split("\\begin{document}", 1)[1]
            for index, chunk in enumerate(split_tutorial_questions(body), start=1):
                # In tutorials \qend carries the total marks, not provenance.
                # A few late-added questions write the marks by hand as
                # \textbf{[6]} instead; read that, then keep it out of the
                # prose so no row leaks its mark count into the text.
                mark_tag = (re.search(r"\\qend\{(\d+)\}", chunk)
                            or re.search(r"\\textbf\{\[(\d+)\]\}(?:\s|\\end\{minipage\})*$", chunk))
                prose_source = re.sub(r"\\hfill\\textbf\{\[\d+\]\}", " ", chunk)
                rows.append({
                    "id": f"tut:{chapter_dir.name}:q{index:02d}",
                    "chapter": chapter,
                    "text": normalise_to_prose(prose_source),
                    "raw_text": chunk,
                    "source": "tutorial_latex",
                    "year": None, "school": None, "level": None,
                    "exam": None, "paper": None, "question_no": None,
                    "modified": False,
                    "marks": int(mark_tag.group(1)) if mark_tag else None,
                })
    return rows


# ---------------------------------------------------------------------------
# Output and verification
# ---------------------------------------------------------------------------

def verify(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate row ids"
    assert all(r["chapter"] in CHAPTER_SLUGS for r in rows), "unknown chapter slug"
    # Solution material must never reach the training set: question rows come
    # only from *_q.tex and non-Solutions files, and their LaTeX never uses
    # the solution macros.
    assert all("\\Sol" not in r["raw_text"] and "\\solend" not in r["raw_text"]
               for r in rows), "solution markup leaked into a question row"

    by_source = collections.Counter(r["source"] for r in rows)
    by_chapter = collections.Counter(r["chapter"] for r in rows)
    print("rows per source:", dict(by_source))
    print("rows per chapter:")
    for slug in CHAPTER_SLUGS:
        print(f"  {by_chapter[slug]:4d}  {slug}")
    print(f"chapters with rows: {len(by_chapter)} of {len(CHAPTER_SLUGS)}")

    shortest = min(rows, key=lambda r: len(r["text"]))
    longest = max(rows, key=lambda r: len(r["text"]))
    print(f"shortest text ({len(shortest['text'])} chars) {shortest['id']}: {shortest['text']}")
    print(f"longest text ({len(longest['text'])} chars) {longest['id']}")

    no_marks = [r["id"] for r in rows if r["marks"] is None]
    if no_marks:
        print("rows without marks:", no_marks)
    if dropped_commands:
        print("unhandled commands dropped (long tail):",
              dict(dropped_commands.most_common()))
    if unparsed_provenance_tokens:
        print("provenance tokens left unclassified:",
              dict(unparsed_provenance_tokens.most_common()))


def main():
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"source root not found: {SOURCE_ROOT}")
    assert_inside_project(OUTPUT_PATH)

    rows = extract_revision_packages() + extract_tutorials()
    verify(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
