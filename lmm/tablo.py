"""YAPILANDIRILMIŞ yutma — tablo (xlsx/csv) → DOĞRUDAN graf + kanıt.

Excel testi dersini kalıcılaştırır: zaten yapılandırılmış kaynakta extractor
(model) kullanmak hem israf hem salınım kaynağı. Tablo yapıyı kendisi söyler:
satır = varlık, sütun = yüklem, hücre = değer. Buradaki yol model ÇAĞIRMAZ →
deterministik, ms — 500 sayfalık tablo bile saniyeler içinde yutulur.

Mimarinin hiçbir kuralı esnetilmez: her üçlü yine KAPIDAN (gate.admit) geçer,
DOCUMENT güveniyle (operatör olgusunu ezemez, cevapta kaynak-çekinceli), satır
cümlesi kanıt deposuna da yazılır (dil yüzeyi oradan kurulur). Başlık sezgisi
xlsx2txt ile aynı: en çok kısa-metin-dolu satır. Belgeye özgü kural yok.
"""
import pandas as pd


def _header_row(rows, columns):
    def fullness(cells):
        return sum(1 for c in cells
                   if isinstance(c, str) and 0 < len(c.strip()) <= 40
                   and not c.startswith("Unnamed"))
    candidates = [(-fullness(r), i) for i, r in enumerate(rows[:10])]
    candidates.append((-fullness(columns), -1))
    candidates.sort()
    return candidates[0][1]


def read_xlsx(path):
    """xlsx → [(sayfa, üstbilgi-satırları, satır-sözlükleri)]. Model yok."""
    sheets = []
    xl = pd.ExcelFile(path)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        rows = df.values.tolist()
        columns = list(df.columns)
        hi = _header_row(rows, columns)
        header = columns if hi == -1 else rows[hi]
        header = [str(h).strip() if not pd.isna(h) else "" for h in header]
        pre = []
        pre_rows = ([columns] if hi >= 0 else []) + (rows[:hi] if hi >= 0 else [])
        for r in pre_rows:
            for cell in r:
                if isinstance(cell, str) and cell.strip() \
                        and not cell.startswith("Unnamed"):
                    pre.append(cell.strip())
        data = []
        for r in (rows[hi + 1:] if hi >= 0 else rows):
            row = {}
            for h, cell in zip(header, r):
                if pd.isna(cell) or not str(cell).strip():
                    continue
                row[h or ""] = str(cell).replace("\n", " ").strip()
            if row:
                data.append(row)
        sheets.append((sheet_name, pre, data))
    return sheets


def learn_xlsx(session, path, source=None):
    """Bir Excel dosyasını yut — extractor'sız, ms. Dönen: yazılan olgu sayısı.
    Üstbilgi 'Anahtar: Değer' satırları da yapısal ayrıştırılır (iki-nokta
    biçim ayracıdır, dil kuralı değil)."""
    source = source or f"#xlsx:{path}"
    wrote = 0
    for sheet, pre, rows in read_xlsx(path):
        for line in pre:
            if ":" in line:
                key, _, value = line.partition(":")
                if key.strip() and value.strip():
                    wrote += session.learn_cell(key.strip(), "", value.strip(),
                                                source)
        wrote += session.learn_rows(rows, source)
        # KANIT katmanı — YALNIZ üstbilgi metni deep=False ile (pencereler
        # üstbilgi satırlarını birbirine bağlar: "Denetleyen ↔ rapor" köprüsü;
        # model çağrısı SIFIR). Veri SATIRLARI pencereye GİRMEZ: kayıt kendi
        # bağlamını zaten taşıyor ve pencere komşu satırların alanlarını
        # birbirine bulaştırıyordu (Ersin'in KKD satırına sızması). Satır
        # cümlelerini learn_rows tekil olarak kanıta yazıyor.
        text = "\n".join([f"[{sheet}]"] + pre)
        session.learn_text(text, source=source, deep=False)
    return wrote
