"""Sohbetin kendisi hakkında sorular: "az önce ne konuşuyorduk".

Graf dünya hakkındaki bilgiyi tutuyor ve tutmalı. Ama bir sohbetin kendisi
dünya hakkında bir olgu değil — "biraz önce penguenlerden konuştuk" cümlesi
grafa yazılırsa hafıza, kalıcı bilgiyle geçici bağlamı karıştırır. Bu ayrım
projenin en baştan beri koruduğu şey: **bilgi kalıcı, bağlam değil.**

O yüzden sohbet geçmişi ayrı bir yerde ve ayrı kurallarla duruyor:

    kalıcı değil    oturum kapanınca gider; grafa hiç dokunmaz
    sınırlı         son birkaç konu tutulur, hepsi değil
    iddia değil     "konuştuk" bir olgu bildirmez, bir kayıt tutar

Neyi çözüyor: on iki soruluk ölçümde "az önce ne konuşuyorduk" hep başarısızdı
ve sebebi bilgi eksikliği değildi — sistemin böyle bir sorusu hiç yoktu.
"""
import collections

KEPT = 12       # kaç konu hatırlanır; sohbet bağlamı kısa olmalı


class Thread:
    """Bu sohbette nelerden konuşulduğu — sırayla, tekrarsız."""

    def __init__(self, kept=KEPT):
        self.kept = kept
        self.topics = collections.deque(maxlen=kept)

    def note(self, concept):
        """Bir kavram konuşuldu. Aynısı üst üste iki kez sayılmaz."""
        if not concept:
            return
        if self.topics and self.topics[-1] == concept:
            return
        if concept in self.topics:
            self.topics.remove(concept)
        self.topics.append(concept)

    def recent(self, count=3):
        """En son konuşulanlar, yeniden eskiye."""
        return list(reversed(self.topics))[:count]

    def __len__(self):
        return len(self.topics)
