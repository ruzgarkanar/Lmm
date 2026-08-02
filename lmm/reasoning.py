"""Reasoning Engine: inheritance, exceptions and conflict detection over memory.

Every answer it produces can name the chain it came from, so nothing the system
says is unexplainable.
"""
from lmm.memory import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY,
                        HAS_PART, LACKS_PART, TYPE_RELATIONS, ABILITY_RELATIONS,
                        PROPERTY_RELATIONS, PART_RELATIONS, INHERITING)
from lmm.phrasing import (ability_clause, property_clause, part_clause,
                          is_a_clause, is_not_a_clause, attribution,
                          disputed_note)
from lmm.trust import INFERENCE, level


def _clause_for(relation):
    """How a relation reads as a sentence. Language, kept out of the reasoning."""
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause
    if relation in (HAS_PART, LACKS_PART):
        return part_clause
    return ability_clause


class Reasoning:
    def __init__(self, memory):
        self.memory = memory

    def ancestors(self, concept):
        """Type ancestors, nearest first, along whichever relation builds them."""
        hierarchy = self.memory.kinds.hierarchical()
        result, queue, seen = [], [concept], {concept}
        while queue:
            current = queue.pop(0)
            for edge in self.memory.query(current, hierarchy):
                if edge.target not in seen:
                    seen.add(edge.target)
                    result.append(edge.target)
                    queue.append(edge.target)
        return result

    def _equivalents(self):
        """Aynı şeyi gösteren adların kümeleri — künyeden okunarak.

        `SAME_AS` künyesi `transitive=True` ve `inverse_of='same_as'` diyor,
        yani bir DENKLİK ilişkisi. İki bayrak da envanterde duruyordu ve motorda
        hiçbir yerden okunmuyordu; ölçüldü:

            otomobil --property--> hızlı
            araba    --same_as---> otomobil
            has_property('araba','hızlı') -> (None, [])

        Yani sistem, kendi kaydettiği eşitliği kullanamıyordu. Bu, "ilişkiler
        veri" iddiasının karşılıksız kaldığı yerdi: yeni bir denklik ilişkisi
        veri olarak bildirilse de hiçbir şey olmuyordu.

        Denklik BURADA da ada bakılarak değil künyeden tanınıyor: geçişli ve
        kendi tersi olan her ilişki bir denkliktir. Yeni bir tane eklenirse bu
        kod değişmeden çalışır.

        Önbellek kenar sayısına bağlı: graf büyürken her sorguda tüm kenarları
        taramak ölçekte kabul edilemez ve tümevarımda tam bu hata ölçülmüştü.
        """
        marker = len(self.memory.edges)
        cached = getattr(self, "_alias_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]
        names = {name for name, kind in self.memory.kinds.by_name.items()
                 if kind.transitive and kind.inverse_of == name}
        groups = {}
        if names:
            for edge in self.memory.edges:
                if edge.relation in names and edge.target:
                    groups.setdefault(edge.concept, set()).add(edge.target)
                    groups.setdefault(edge.target, set()).add(edge.concept)
        self._alias_cache = (marker, groups)
        return groups

    def aliases(self, concept):
        """Bu kavramla aynı şeyi gösteren diğer adlar, geçişli kapanışıyla."""
        groups = self._equivalents()
        if concept not in groups:
            return []
        found, queue, seen = [], [concept], {concept}
        while queue:
            for other in sorted(groups.get(queue.pop(0), ())):
                if other not in seen:
                    seen.add(other)
                    found.append(other)
                    queue.append(other)
        return found

    def about(self, concept, relation, target, object=None, role=None):
        """Any relation at all, read from the registry rather than a branch.

        This is what lets a new kind of fact arrive as data: nothing here knows
        what "has" means, only that it denies "has_not" and carries down.
        """
        affirms, denies = self.memory.kinds.pair(relation)
        clause = _clause_for(relation)
        answer, chain, _ = self._resolve(concept, target, affirms, denies,
                                         clause, object, role)
        return answer, chain

    def can_do(self, concept, action, object=None, role=None):
        """Returns (True | False | None, explanation chain).

        None means memory holds nothing on this — the caller must not guess.
        """
        answer, chain, _ = self._resolve(concept, action, CAN, CANNOT,
                                         ability_clause, object, role)
        return answer, chain

    def has_property(self, concept, prop, object=None, role=None):
        """Same shape as can_do, for "kar beyazdır" style knowledge."""
        answer, chain, _ = self._resolve(concept, prop, HAS_PROPERTY,
                                         LACKS_PROPERTY, property_clause,
                                         object, role)
        return answer, chain

    def basis(self, candidate):
        """The edge behind the current belief about a candidate's claim.

        Lets a caller ask *why* it believes something — in particular whether a
        belief came from a teacher or from the system's own generalisation.
        """
        if candidate.relation in ABILITY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target, CAN,
                                 CANNOT, ability_clause, candidate.object,
                                 candidate.role)[2]
        if candidate.relation in PROPERTY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target,
                                 HAS_PROPERTY, LACKS_PROPERTY, property_clause,
                                 candidate.object, candidate.role)[2]
        return self.memory.direct(candidate.concept,
                                  IS_A if candidate.relation == NOT_A else NOT_A,
                                  candidate.target)

    def _resolve(self, concept, target, affirms, denies, clause, object=None,
                 role=None):
        """Direct knowledge first, then the type hierarchy, nearest ancestor first.

        A direct fact always beats an inherited one — that is exactly what makes
        an exception an exception.
        """
        # Both polarities may be on record when sources disagreed. A settled
        # claim outranks one still marked disputed, so arbitration actually
        # changes the answer instead of leaving the loser to speak first. Among
        # equally settled claims the stronger voice speaks: a person correcting
        # a document was being accepted, thanked, and then ignored, because the
        # document's edge simply came first in the list.
        held = [(polarity, self.memory.direct(concept, relation, target, object,
                                              role))
                for polarity, relation in ((False, denies), (True, affirms))]
        held = [(polarity, edge) for polarity, edge in held if edge]
        held.sort(key=lambda pair: (pair[1].disputed, -level(pair[1].source)))
        if held:
            polarity, edge = held[0]
            said = clause(concept, target, polarity, object, role)
            note = disputed_note() if edge.disputed else attribution(edge.source)
            return polarity, [f"{said} ({note})"], edge
        # Denk adlar, ATALARDAN ÖNCE. "araba" ile "otomobil" aynı şeyse
        # otomobil hakkında bilinen doğrudan bilgidir; atadan miras değil.
        # Zincirdeki bağ `=` ile yazılıyor: bu dosyada Türkçe metin biriktirmek
        # mimari ihlali ve `=` her dilde aynı şeyi söylüyor.
        for other in self.aliases(concept):
            for polarity, relation in ((False, denies), (True, affirms)):
                edge = self.memory.direct(other, relation, target, object, role)
                if edge:
                    said = clause(other, target, polarity, object, role)
                    note = (disputed_note() if edge.disputed
                            else attribution(edge.source))
                    return polarity, [f"{concept} = {other}",
                                      f"{said} ({note})"], edge
        for ancestor in self.ancestors(concept):
            for polarity, relation in ((False, denies), (True, affirms)):
                edge = self.memory.direct(ancestor, relation, target, object, role)
                if edge and not self.memory.kinds.inherits(relation):
                    continue        # some relations simply do not carry down
                if edge and edge.quantifier not in INHERITING:
                    # "bazı kuşlar uçmaz" says nothing about this bird. Letting
                    # it inherit would turn an existence claim into a universal.
                    continue
                if edge:
                    inherited = clause(ancestor, target, polarity, object, role)
                    if edge.source == INFERENCE:
                        # An inherited guess is still a guess, and must say so.
                        inherited += " (kendi çıkarımım)"
                    return polarity, [f"{concept} bir {ancestor}", inherited], edge
        return None, [], None

    def abilities(self, concept):
        """[(action, True|False)] for every action this memory knows about."""
        return self._known_of(self.memory.actions(), self.can_do, concept)

    def properties(self, concept):
        """[(property, True|False)] for every property this memory knows about."""
        return self._known_of(self.memory.properties(), self.has_property, concept)

    def _known_of(self, targets, lookup, concept):
        found = []
        for target in targets:
            known, _ = lookup(concept, target)
            if known is not None:
                found.append((target, known))
        return found

    def who_can(self, action, positive=True):
        """Every concept known to do (or known not to do) an action."""
        return [c for c in self.memory.concepts()
                if self.can_do(c, action)[0] is positive]

    def survey(self, concept, target, relation_pair):
        """Which members of a kind are known to do this, and which are not.

        "Bazı kuşlar uçmaz" does not have to be stored to be answered: the
        exceptions already on record say it. An existence claim is a reading of
        the memory, not another fact in it.
        """
        lookup = (self.has_property if relation_pair[0] == HAS_PROPERTY
                  else self.can_do)
        yes, no = [], []
        for other in self.memory.concepts():
            if other == concept or concept not in self.ancestors(other):
                continue
            # Inherited counts: a sparrow flies because it is a bird, and the
            # question is about the members, not about who was told what.
            known, _ = lookup(other, target)
            if known is True:
                yes.append(other)
            elif known is False:
                no.append(other)
        return yes, no

    def find_conflict(self, candidate):
        """Explanation string if the candidate edge conflicts with what we know."""
        if candidate.relation in TYPE_RELATIONS:
            return self._type_conflict(candidate)
        affirms, denies = self.memory.kinds.pair(candidate.relation)
        if denies is None:
            return None
        known, chain = self.about(candidate.concept, candidate.relation,
                                  candidate.target, candidate.object,
                                  candidate.role)
        claimed = candidate.relation == affirms
        if known is not None and known != claimed:
            return "şu an bildiğim: " + " çünkü ".join(chain)
        return None

    def _type_conflict(self, candidate):
        """"penguen bir kuş değildir" against a hierarchy that says it is."""
        opposite = NOT_A if candidate.relation == IS_A else IS_A
        edge = self.memory.direct(candidate.concept, opposite, candidate.target)
        if edge is not None:
            clause = (is_a_clause if opposite == IS_A else is_not_a_clause)
            return (f"şu an bildiğim: {clause(candidate.concept, candidate.target)} "
                    f"(kaynak: {edge.source})")
        if (candidate.relation == NOT_A
                and candidate.target in self.ancestors(candidate.concept)):
            return ("şu an bildiğim: "
                    + is_a_clause(candidate.concept, candidate.target))
        return None
