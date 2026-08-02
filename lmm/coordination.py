"""İki özneli sorular: "penguen ve kartal ikisi de kuş mu".

Ölçümde başarısız olan sorulardan biriydi ve sebebi bilgi eksikliği değildi:
graf hem penguenin hem kartalın kuş olduğunu biliyor. Eksik olan, **bir cümlede
iki özne** olabileceğiydi. Ayrıştırıcı "penguen ve kartal" öbeğini tek bir
kavram sanıyor, grafta öyle bir şey bulamıyor ve soru sessizce düşüyordu.

Çözüm bir kalıp eklemek değil — çünkü her soru türü için ikinci bir "iki özneli"
kalıp yazmak, kalıp sayısını ikiye katlar ve üç özneli soruda yine çöker.
Bunun yerine cümle **bölünüyor**: bağlaçla ayrılmış her özne için aynı soru
sorulur, cevaplar birleştirilir.

    "penguen ve kartal kuş mu"
      -> "penguen kuş mu"  + "kartal kuş mu"
      -> ikisi de evet     -> "evet, ikisi de"

Bu, `lmm/clauses.py`'nin cümleleri bölerken yaptığı işin özne düzeyindeki
karşılığı. Ve aynı disipline uyuyor: bölünmüş parçaların her biri normal
yoldan, kapıdan geçerek cevaplanıyor — birleştirme yalnızca söyleyişte.

"ikisi de", "her ikisi" gibi sözcükler atılabilir: bilgi taşımıyorlar, zaten
iki özne olduğunu bağlaç söylüyor.
"""
from lmm import phrasing

# Cümlede iki özne olduğunu bağlaç zaten söylüyor; bu sözcükler onu tekrar
# ediyor ve ayrıştırmayı bozuyorlar. Kapalı sınıf, birkaç tane.
ECHOES = ("ikisi", "ikiside", "hepsi", "her", "ikisininde", "de", "da")


def split_subjects(tokens, joiners, known):
    """Bağlaçla ayrılmış özneler ve cümlenin geri kalanı.

    (özneler, kalan) döner; iki özne bulunamazsa (None, None).

    Özne sayılmak için grafta tanınmak gerekiyor. Bu bilinçli: "ali ve
    veli geldi" cümlesinde tanınmayan iki ad varsa, onları özne diye
    ayırmak uydurma olurdu — bilinmeyen bir şey hakkında soru zaten
    cevaplanamaz.
    """
    at = next((i for i, token in enumerate(tokens) if token in joiners), None)
    if at is None or at == 0 or at == len(tokens) - 1:
        return None, None
    first = tokens[:at]
    if len(first) != 1 or first[0] not in known:
        return None, None
    rest = tokens[at + 1:]
    if not rest or rest[0] not in known:
        return None, None
    second, tail = rest[0], rest[1:]
    tail = [token for token in tail if token not in ECHOES]
    if not tail:
        return None, None
    return [first[0], second], tail


def combine(answers):
    """İki cevabı tek cümlede birleştirir — aynıysa kısaltarak.

    Kaç cevap olduğu bir yapı sorusu ve burada kalıyor; iki cevabın Türkçede
    nasıl tek cümle olduğu bir dil sorusu ve `lmm/phrasing.py`'ye taşındı.
    "evet, ikisi de" Türkçenin kısaltması, birleştirmenin kendisi değil.
    """
    if not answers:
        return None
    if len(answers) == 1:
        return answers[0]
    return phrasing.combined(answers)
