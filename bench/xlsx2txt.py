"""GENEL xlsx→metin dönüştürücü — LMM yutması için. Belgeye özgü kural YOK:
her sayfada başlık satırı sezgisel bulunur (en çok dolu-metin hücreli satır),
her veri satırı "Başlık: değer · Başlık: değer" cümlesine serileştirilir —
böylece hücre-değeri hangi niteliğe ait olduğunu YANINDA taşır (tablo-bağlama
sorunu kaynağında çözülür). Başlık-öncesi satırlar düz metin olarak geçer.

Kullanım: python3.11 bench/xlsx2txt.py dosya.xlsx > çıktı.txt
"""
import sys

import pandas as pd


def sheet_to_text(df, sheet_name):
    out = [f"[{sheet_name}]"]
    rows = df.values.tolist()
    columns = list(df.columns)
    # pandas ilk satırı başlık sanmış olabilir; gerçek başlığı içeride ara:
    # en çok kısa-metin-dolu hücresi olan satır.
    def fullness(cells):
        return sum(1 for c in cells
                   if isinstance(c, str) and 0 < len(c.strip()) <= 40
                   and not c.startswith("Unnamed"))
    candidates = [(-fullness(r), i) for i, r in enumerate(rows[:10])]
    candidates.append((-fullness(columns), -1))     # pandas başlığı da aday
    candidates.sort()
    header_i = candidates[0][1]
    header = columns if header_i == -1 else rows[header_i]
    header = [str(h).strip() if not pd.isna(h) else "" for h in header]
    # başlık-öncesi hücreler: düz metin (rapor üstbilgisi vb.)
    pre = rows[:header_i] if header_i >= 0 else []
    if header_i >= 0 and list(columns) and not all(
            str(c).startswith("Unnamed") for c in columns):
        pre = [columns] + pre
    for r in pre:
        for cell in r:
            if isinstance(cell, str) and cell.strip() \
                    and not cell.startswith("Unnamed"):
                out.append(cell.strip())
    # veri satırları: "Başlık: değer · ..."
    data = rows[header_i + 1:] if header_i >= 0 else rows
    for r in data:
        parts = []
        for h, cell in zip(header, r):
            if pd.isna(cell) or not str(cell).strip():
                continue
            value = str(cell).replace("\n", " ").strip()
            parts.append(f"{h}: {value}" if h else value)
        if parts:
            out.append(" · ".join(parts) + ".")
    return "\n".join(out)


def main():
    xl = pd.ExcelFile(sys.argv[1])
    print("\n\n".join(sheet_to_text(xl.parse(sn), sn)
                      for sn in xl.sheet_names))


if __name__ == "__main__":
    main()
