"""What LMM does with NO language model at all.

    python examples/no_engine.py

Every other example needs an engine, because turning a question into a sentence
is a language model's job. This one needs nothing: the graph, the gate, the
trust ordering and the derivation are pure python, and they are most of what
makes the system's answers checkable. Structured ingestion reaches them without
a model call, which is why a spreadsheet loads in milliseconds.
"""
import csv
import os
import tempfile

from lmm import Memory

folder = tempfile.mkdtemp()
path = os.path.join(folder, "fleet.csv")
with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["probe", "built_on", "range", "power"])
    writer.writerow(["Norgul", "Nortlann", "5 m", "40 W"])
    writer.writerow(["Zerbal", "Vorlin", "8 m", "65 W"])

m = Memory()
report = m.learn(path)
print(f"{report.facts} facts from {report.tables} table, "
      f"via the {report.adapter} reader — zero model calls")

# The graph, directly. Records carry integer identity keys rather than strings:
# the same spelling can name two concepts, so an identity is not its label.
memory = m.session.memory
for record in m.about("Norgul"):
    field = memory.identities[record.predicate].labels[0]
    value = memory.identities[record.value].labels[0]
    print(f"  Norgul · {field} · {value}   (source {record.source},"
          f" trust {record.trust})")

print(f"\n{len(m)} facts. Nothing here loaded a model.")
