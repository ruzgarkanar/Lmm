"""Kurallı ayrıştırıcı, öğrenilmiş etiketleyiciyle AYNI sınavda.

Kıyas olmadan "%52 kavram" bir şey söylemiyor. Aynı 3.000 tutulmuş cümle,
aynı ölçüt: sistemin kendi ayrıştırıcısı kavramı ve hedefi doğru buluyor mu.
"""
import os, random, shutil, sys, tempfile
sys.path.insert(0,"/Users/ruzgarkanar/Desktop/MyBOT"); os.chdir("/Users/ruzgarkanar/Desktop/MyBOT")
from core.tagger import rows_from, spans_from
from lmm.cli import Session

paths=[a for a in sys.argv[1:] if a.endswith(".jsonl")]
rows=rows_from(paths)
random.Random(7).shuffle(rows)
exam=rows[:3000]                     # eğitimin hiç görmediği, aynı tohum
W=os.path.join(tempfile.mkdtemp(),"k.lmm"); shutil.copy("models/graph/birlesik.lmm",W)
s=Session(W); lang=s.language
right=both=read=0
for sentence, marks, _ in exam:
    want_c, want_t = spans_from(sentence, marks)
    it = lang.understand(sentence)
    if it.concept:
        read += 1
        got_c = str(it.concept)
        got_t = str(it.target or "")
        if got_c.lower() == want_c.lower(): right += 1
        if got_c.lower()==want_c.lower() and got_t.lower()==want_t.lower(): both += 1
print(f"  KURALLI AYRIŞTIRICI, {len(exam):,} tutulmuş cümlede:")
print(f"    bir kavram çıkardı  %{read/len(exam)*100:.1f}")
print(f"    KAVRAM doğru        %{right/len(exam)*100:.1f}")
print(f"    KAVRAM+HEDEF doğru  %{both/len(exam)*100:.1f}")
