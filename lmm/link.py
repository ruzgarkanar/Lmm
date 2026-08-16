"""Etiket → kimlik köprüsü. Qwen'in çıkardığı metin etiketlerini grafın sayısal
kimliklerine çevirir. `geometry.resolve` çokanlamlılığı bağlamla çözer (aynı
yazılış birden çok kavram olabilir); yoksa `memory.identify` açar.

KAYIT YAZMAZ — yalnız etiket↔kimlik eşler. Neyin yazılacağına KAPI karar verir.
Büyük/küçük harf bağımsız (`fold`), dil kuralı yok.
"""
from v3 import geometry
from v3.dataset import fold


def resolve(memory, label, vectors=None, create=False):
    """Etiketi kimliğe çevirir.

    create=False: yalnız VAR OLAN kimliği döner, yoksa None (doğrulama için —
                  Qwen yeni ad uydurduysa karşılığı yok, iddia desteksiz sayılır).
    create=True:  yoksa yeni kimlik açar (yazma için).
    """
    if not label:
        return None
    label = fold(str(label).strip())
    if not label:
        return None
    vec = (vectors or {}).get(label)
    key, _ = geometry.resolve(memory, label, vec)
    if key is not None:
        return key
    # GÜVENLİ ÇEKİM TOLERANSI: Qwen çekimli verebilir ("organ"→"organdır").
    # Ölçüt SIKI ki F5 deliği (kart→kartal, organ→organizma) açılmasın:
    #   kök >=5 harf VE artık <=3 harf.
    #     organ(5)→organdır  artık "dır"=3  ✓ eşleşir (doğru)
    #     kart(4)→kartal      kök 4<5        ✗ (yanlış-eşleşme kapalı)
    #     organ(5)→organizma  artık "izma"=4 ✗ (yanlış-eşleşme kapalı)
    # "Uydurma 0" için sıkı taraf: gevşetip yanlış-pozitif almıyoruz.
    for other, ident in memory.identities.items():
        for lab in ident.labels:
            root = fold(lab)
            if len(root) >= 5 and label.startswith(root) \
                    and 0 < len(label) - len(root) <= 3:
                return other
            # TERS YÖN — YALNIZ YAZARKEN (create=True). Var olan düğüm çekimli
            # açılmış olabilir ("metaldir" önce değer olarak geldi), sonra kök
            # gelir ("metal" özne) — tek yönlü bakış iki düğüm açıp geçişli
            # zinciri koparıyordu. AMA doğrulama/okuma yolunda (create=False)
            # bu eşleme KAPALI: grafta yalnız "şekersiz" varken Qwen'in "şeker
            # ..." iddiası yanlış düğümün kenarıyla "destekli" sayılırdı —
            # uydurma-0 deliği (code-review bulgusu #1). Yazarken eşleşince kök
            # ALIAS olarak kimliğe eklenir; sonraki okumalar kör tarama değil,
            # meşru etiket eşleşmesiyle bulur. F5 korumaları aynen.
            if create and len(label) >= 5 and root.startswith(label) \
                    and 0 < len(root) - len(label) <= 3:
                return memory.identify(label, same_as=other)
    if create:
        return memory.identify(label, vector=vec)
    return None


def label_of(memory, key):
    """Kimlikten ilk etikete (ham döküm / prompt için). None → boş."""
    if key is None:
        return ""
    held = memory.identities.get(key) if isinstance(key, int) else None
    if held is not None and held.labels:
        return held.labels[0]
    return str(key)
