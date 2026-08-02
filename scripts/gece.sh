#!/bin/zsh
# Gece hattı: etiketle -> eğit -> ölç -> daha çok etiketle -> yeniden eğit.
#
# Her adım diske yazar ve bir sonraki adım ondan okur; biri düşerse hat durmaz.
# Ölçüt hep aynı: 280 gerçek soruluk ayrılmış sınav, ve gecenin tabanı %66,4.
set -o pipefail
cd /Users/ruzgarkanar/Desktop/MyBOT
set -a; . /tmp/lmm_env.sh 2>/dev/null; set +a
LOG=/tmp/gece_rapor.log
say() { echo "\n=== $(date +%H:%M) $1" | tee -a $LOG; }

# --- 1. dalga etiketleme zaten dönüyor; bitmesini bekle -------------------
say "1. DALGA ETİKETLEME bekleniyor"
while pgrep -f aday_etiketle.py >/dev/null; do sleep 60; done
tail -12 /tmp/aday.log | tee -a $LOG

# --- veriyi birleştir ------------------------------------------------------
say "VERİ BİRLEŞTİRME"
python3 - <<'PY' 2>&1 | tee -a $LOG
import collections, os
ROOT="/Users/ruzgarkanar/Desktop/MyBOT"
kaynak=["data/tr-gercek-niyet-buyuk.txt","data/tr-gercek-niyet.txt","data/tr-niyet-genis.txt"]
gorulen=set(); satir=[]
for k in kaynak:
    p=os.path.join(ROOT,k)
    if not os.path.exists(p): continue
    n=0
    for l in open(p,encoding="utf-8"):
        if "\t" not in l: continue
        tur,c=l.rstrip("\n").split("\t",1)
        if c in gorulen: continue
        gorulen.add(c); satir.append((tur,c)); n+=1
    print(f"  {k}: {n} yeni")
with open(os.path.join(ROOT,"data/tr-niyet-tumu.txt"),"w",encoding="utf-8") as o:
    for t,c in satir: o.write(f"{t}\t{c}\n")
s=collections.Counter(t for t,_ in satir)
print(f"  TOPLAM {len(satir)} ayrı soru")
ici=sum(v for k,v in s.items() if k!="DISARIDA")
for k,v in s.most_common(): print(f"    {k:<18} {v}")
print(f"  kapsam içi: {ici}")
PY

# --- 2. eğitim: kendi çekirdeğimiz, birkaç ayarla ---------------------------
for hiz in 8e-5 5e-5 1.5e-4; do
  say "EĞİTİM çekirdek hiz=$hiz"
  python3.11 scripts/niyet_egitim.py --omurga cekirdek --tur 5 --hiz $hiz \
    --yigin 64 --veri data/tr-niyet-tumu.txt 2>&1 | grep -E "veri:|TUR|en iyi|->" | tee -a $LOG
done

# --- 3. ikinci dalga aday + etiketleme --------------------------------------
say "2. DALGA SÜZGEÇ"
python3.11 scripts/soru_suz.py \
  --girdi /private/tmp/claude-501/-Users-ruzgarkanar-Desktop-MyBOT/f6f06f65-c1a2-456a-8faf-41ceecaddff9/scratchpad/tum_sorular.json \
  --sayi 60000 --tara 1500000 --atla 200000 2>&1 | tail -12 | tee -a $LOG

say "2. DALGA ETİKETLEME"
python3 scripts/aday_etiketle.py --sayi 60000 --cikti data/tr-niyet-genis2.txt 2>&1 | tail -12 | tee -a $LOG

# --- 4. hepsiyle son eğitim -------------------------------------------------
say "SON BİRLEŞTİRME"
python3 - <<'PY' 2>&1 | tee -a $LOG
import collections, os
ROOT="/Users/ruzgarkanar/Desktop/MyBOT"
gorulen=set(); satir=[]
for k in ["data/tr-niyet-tumu.txt","data/tr-niyet-genis2.txt"]:
    p=os.path.join(ROOT,k)
    if not os.path.exists(p): continue
    for l in open(p,encoding="utf-8"):
        if "\t" not in l: continue
        t,c=l.rstrip("\n").split("\t",1)
        if c in gorulen: continue
        gorulen.add(c); satir.append((t,c))
with open(os.path.join(ROOT,"data/tr-niyet-tumu.txt"),"w",encoding="utf-8") as o:
    for t,c in satir: o.write(f"{t}\t{c}\n")
s=collections.Counter(t for t,_ in satir)
print(f"  TOPLAM {len(satir)}  kapsam içi {sum(v for k,v in s.items() if k!='DISARIDA')}")
PY

say "SON EĞİTİM çekirdek"
python3.11 scripts/niyet_egitim.py --omurga cekirdek --tur 6 --hiz 8e-5 --yigin 64 \
  --veri data/tr-niyet-tumu.txt 2>&1 | grep -E "veri:|TUR|en iyi|->" | tee -a $LOG

say "SON EĞİTİM 355M"
python3 scripts/niyet_egitim.py --omurga hazir --tur 2 --yigin 16 \
  --veri data/tr-niyet-tumu.txt 2>&1 | grep -E "veri:|TUR|en iyi|->" | tee -a $LOG

say "HAT BİTTİ"
