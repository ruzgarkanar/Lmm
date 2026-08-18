"""The whole library, in five lines.

    python examples/quickstart.py

Needs an engine — LMM's graph and gate are pure python, but turning a question
into a sentence is a language model's job. See the README's "Engines" section;
the default is a local Qwen2.5-3B-Instruct.
"""
from lmm import Memory

m = Memory("mind.lmm")                     # persistent; Memory() is transient
m.learn("The Norgul probe was built on Nortlann in 2031.")
print(m.ask("where was the Norgul probe built?"))
m.save()
