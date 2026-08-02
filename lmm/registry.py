"""Hangi model dosyası nerede, hangisi güncel — tek yerden.

Model dosyaları kök dizine dağılmıştı: `lmm-cekirdek.pt`, `cekirdek.pt`,
`niyet.pt`, `model.lmm`, `model-wiki.lmm`. Hangisinin eski hangisinin yeni
olduğu adından anlaşılmıyordu ve yollar sekiz ayrı dosyaya elle yazılmıştı.
Yeni bir sürüm eğitildiğinde sekiz yerin de değişmesi gerekiyordu.

Burada iki şey ayrılıyor:

    sürüm       dosyanın kendisi, adında ne olduğu yazılı (97m-30k.pt)
    güncel      hangisinin kullanılacağı — künyede yazıyor, dosya adında değil

Böylece yeni bir sürüm eskisinin üzerine yazılmıyor; yan yana duruyor ve
künyedeki tek satır değiştirilerek geçiş yapılıyor. Kötü çıkarsa geri dönmek
de aynı tek satır.

Künye (`models/registry.json`) ayrıca her sürümün **nasıl ölçüldüğünü** taşıyor.
Bir modelin skoru dosyanın yanında durmazsa, altı ay sonra hangisinin neden
seçildiği bilinmez olur.

Bağımlılık yok: yalnız `json` ve `os`. `lmm/` sıfır bağımlılık kuralı bozulmuyor.
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(HERE, "models")
MANIFEST = os.path.join(FOLDER, "registry.json")

# Künye yoksa ya da bozuksa bunlar kullanılır. Sistem, künye olmadan da
# çalışmalı — kayıt bir kolaylık, bir bağımlılık değil.
FALLBACK = {
    "core": os.path.join(FOLDER, "core", "16m-8k.pt"),
    "intent": os.path.join(FOLDER, "intent", "16m.pt"),
    "classes": os.path.join(FOLDER, "intent", "classes.json"),
    "graph": os.path.join(FOLDER, "graph", "base.lmm"),
}

_loaded = None


def manifest(path=MANIFEST):
    """Künyenin tamamı. Yoksa boş sözlük."""
    global _loaded
    if _loaded is None:
        try:
            with open(path, encoding="utf-8") as handle:
                _loaded = json.load(handle)
        except Exception:                                   # noqa: BLE001
            _loaded = {}
    return _loaded


def where(kind, name=None):
    """Bu türden modelin yolu. `name` verilmezse künyedeki güncel sürüm.

    Dosya yoksa None döner — çağıran bunu bilir ve sessizce yanlış dosya
    yüklemez.
    """
    found = manifest().get(kind, {})
    chosen = name or found.get("current")
    versions = found.get("versions", {})
    path = versions.get(chosen, {}).get("path") if chosen else None
    if path:
        path = os.path.join(FOLDER, path) if not os.path.isabs(path) else path
    else:
        path = FALLBACK.get(kind)
    return path if path and os.path.exists(path) else None


def about(kind, name=None):
    """Bu sürüm hakkında bilinenler: ne zaman, hangi veriyle, hangi skor."""
    found = manifest().get(kind, {})
    chosen = name or found.get("current")
    return found.get("versions", {}).get(chosen, {}) if chosen else {}


def versions(kind):
    """Bu türden elde ne varsa — adları."""
    return sorted(manifest().get(kind, {}).get("versions", {}))
