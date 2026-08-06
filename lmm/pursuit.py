"""Pursuit: holding a question it cannot answer, and working out what it needs.

Curiosity wanders — it notices any gap. Pursuit is aimed: given one question the
system cannot answer, it derives from memory the single fact that would unlock
it, asks for that, and comes back to the original question the moment it can.

This is the agent loop without an agent guessing at tools. The plan is not
invented; it is read off the hierarchy. "I don't know whether a penguin flies,
but I know a penguin is a bird — do birds fly?" is a step the memory itself
dictates.
"""
from lmm.relations import IS_A, CAN, HAS_PROPERTY, HAS_PART, LACKS_PART
from lmm import serialize
from lmm.similarity import nearest


class Step:
    def __init__(self, key, text):
        self.key = key      # stable identity, so the same step is not repeated
        self.text = text


class Pursuit:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def resolved(self, goal):
        """Can the gate answer this question from what memory holds now?"""
        if goal.relation == IS_A:
            return bool(self.memory.query(goal.concept, IS_A))
        return self._lookup(goal) is not None

    def next_step(self, goal):
        """The one thing worth asking next, or None when nothing would help."""
        if self.resolved(goal):
            return None
        if not goal.concept:
            return None
        if goal.relation == IS_A or not self.memory.query(goal.concept, IS_A):
            # Without a place in the hierarchy there is nothing to reason from.
            return Step(f"type:{goal.concept}",
                        f"{goal.concept} → {serialize.label(IS_A)} → ?")
        # Hedefsiz bir amacın ARA SORUSU olamaz: "kartal nasıldır" belirli bir
        # nitelik sormuyor, dolayısıyla "atası o niteliği taşıyor mu" diye
        # sorulacak bir şey de yok. Denetim yoktu ve hedefsiz amaç
        # `phrasing.property_question(None)` çağırıp sistemi ÇÖKERTİYORDU.
        #
        # Ortaya çıkışı öğreticiydi: yeni kalıplar bağlanınca hedefsiz okumalar
        # ilk kez bu yola geldi — bir yeteneğin başka bir kusuru açığa
        # çıkarması, bugün üçüncü kez.
        #
        # Denetim buraya kondu, işlevin başına DEĞİL: tanım sorusu da
        # hedefsizdir ve o yukarıdaki dalla cevaplanıyor. Başa konunca yazım
        # yanlışı önerisi ("kuş mu demek istedin") sessizce kayboldu.
        if not goal.target:
            return None
        for ancestor in self.reasoning.ancestors(goal.concept):
            if self._lookup(goal, ancestor) is None:
                return Step(f"{goal.relation}:{ancestor}:{goal.target}",
                            self._question(ancestor, goal))
        return None

    def opening(self, goal):
        """What to say when a question arrives that cannot be answered yet."""
        step = self.next_step(goal)
        if step is None:
            return f"[?] {goal.concept}"
        if step.key.startswith("type:"):
            # A concept with nothing at all behind it might be a word the person
            # spelled differently from the one we know.
            suggestions = ()
            if not self.memory.query(goal.concept):
                suggestions = nearest(goal.concept, self.memory.concepts())
                # Anlam komşusu yazım komşusundan ÖNCE gelir. İlk yazışta
                # yalnız yazım komşusu yokken bakıyordum ve neredeyse hiç
                # çalışmıyordu: `nearest` çoğu kelimede harfçe benzer bir şey
                # buluyor ("glokom" -> "glikoz"), ama harf benzerliği anlam
                # değildir ve kullanıcıya yardımı yok. Anlamca yakın BİLİNEN
                # bir kavram varsa o söylenir.
                related = self._related_known(goal.concept)
                if related:
                    return f"[?] {goal.concept} · ≈ {', '.join(related)}"
                if not suggestions:
                    # Yazım komşusu yoksa ANLAM komşusu. İkisi ayrı şey: "kus"
                    # ile "kuş" harf komşusu, "glokom" ile "katarakt" anlam
                    # komşusu ve ikincisini yalnız dağılım verebilir.
                    #
                    # Ölçüldü: 400 gerçek sorunun %56,1'inde grafın hiç
                    # duymadığı bir kavram var ve çoğunda yazım komşusu da yok,
                    # yani cevap düz "bilmiyorum" oluyordu.
                    #
                    # Söylenen bir İDDİA DEĞİL: sorulan kavram hakkında hiçbir
                    # şey söylenmiyor, elde ne olduğu gösteriliyor. Vektör
                    # hiçbir zaman olgu üretmiyor.
                    related = self._related_known(goal.concept)
                    if related:
                        return f"[?] {goal.concept} · ≈ {', '.join(related)}"
            plain = f"[?] ⊢ {step.text}"
            if suggestions:
                plain += f" ≈ {', '.join(list(suggestions))} ?"
            return plain
        ancestors = self.reasoning.ancestors(goal.concept)
        return f"[?] {serialize.fact(goal.concept, IS_A, ancestors[0])} · {step.text}"

    def _related_known(self, concept, count=3):
        """Anlamca yakın ve grafın BİLDİĞİ kavramlar — vektör varsa."""
        vectors = getattr(self.memory, "vectors", None)
        if vectors is None or not concept:
            return ()
        # Yapı sözcüğünden komşu çıkmaz: "turizmle aynı" ifadesinin vektörü
        # bileşimde `aynı`ya çöküyor ve `aynı` her yerde geçtiği için
        # komşuları gürültü ("scobey", "kartuş"). Ölçüt sıra, liste değil.
        from lmm import frequency
        from lmm.verbs import is_structural
        counts = frequency.counts()
        meaningful = [word for word in concept.split()
                      if not is_structural(word, counts)]
        if not meaningful:
            return ()
        try:
            close = vectors.similar(" ".join(meaningful), 40)
        except Exception:                                   # noqa: BLE001
            return ()
        known = set(self.memory.concepts())
        found = []
        for word, score in close:
            # Eşiğin altı gürültü: "ilgili olabilir" diye sunulan şey
            # gerçekten yakın olmalı, yoksa öneri sistemi rastgele gösterir.
            if score < 0.45:
                break
            if word in known and word != concept and self.memory.query(word):
                found.append(word)
            if len(found) >= count:
                break
        return tuple(found)

    def _lookup(self, goal, concept=None):
        concept = concept if concept is not None else goal.concept
        obj = getattr(goal, "object", None)
        role = getattr(goal, "role", None)
        return self.reasoning.about(concept, goal.relation, goal.target,
                                    obj, role)[0]

    def _question(self, concept, goal):
        obj = getattr(goal, "object", None)
        role = getattr(goal, "role", None)
        if goal.relation in (HAS_PART, LACKS_PART):
            return f"{serialize.fact(concept, HAS_PART, goal.target)} ?"
        if goal.relation == HAS_PROPERTY:
            return f"{serialize.fact(concept, HAS_PROPERTY, goal.target, True, obj)} ?"
        return f"{serialize.fact(concept, CAN, goal.target, True, obj)} ?"
