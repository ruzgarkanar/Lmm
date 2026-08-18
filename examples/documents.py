"""Teaching LMM real files: PDF, Word, Excel, CSV, Markdown.

    pip install 'lmm[pdf,xlsx,docx]'
    python examples/documents.py manual.pdf spec.docx inspection.xlsx

One method reads all of them. The format is taken from the extension, and the
adapter behind it decides what goes where: a document is not one material but
two, and they want opposite treatment.

    TABLES  ->  straight into the graph as facts. No model call, milliseconds
                even for a large sheet, and the row keeps its structure —
                flattening a spec table into prose is how the value in a cell
                stops being attached to the field that names it.

    PROSE   ->  the evidence layer, where the numbers, ranges and qualifications
                that do not fit a triple survive as sentences.

`deep` decides whether the ENGINE also mines each sentence for triples. It
defaults to False for files on purpose: a 350-page manual's structure is already
taken by the adapters, and running an extractor over every sentence would cost
minutes to add little. Pass deep=True for a short, dense document you want
fully in the graph.
"""
import sys

from lmm import Memory


def main(paths):
    m = Memory("documents.lmm")

    for path in paths:
        try:
            report = m.learn(path)
        except ImportError as missing:
            # The core install is dependency-free; each reader asks for itself.
            print(f"  {path}: {missing}")
            continue
        except (ValueError, OSError) as refused:
            print(f"  {path}: {refused}")
            continue
        print(f"  {path}: {report.facts} facts, {report.tables} tables "
              f"via the {report.adapter} reader, stamped {report.source}")

    print(f"\n{len(m)} facts in memory\n")

    # Ask with explain=True and the answer reports itself: the sources it
    # rests on, and whether it asserted anything at all. An abstention is a
    # RESULT here, not a failure — the gate is what makes "I don't know"
    # trustworthy.
    for question in ("what is the screen's diagonal?",
                     "what is the warranty period?"):
        answer = m.ask(question, explain=True)
        mark = "abstained" if answer.abstained else "answered"
        print(f"Q {question}\nA [{mark}] {answer}")
        if answer.sources:
            print(f"  sources: {', '.join(answer.sources)}")
        print()

    m.save()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
