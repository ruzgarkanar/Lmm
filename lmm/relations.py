"""The kinds of link a fact can be.

They live in their own module because everything depends on them — memory,
reasoning, language and phrasing alike — and nothing should have to depend on
memory just to name a relation.

Each pair is one affirmation and its denial. Reasoning treats every pair the
same way: a direct fact wins over an inherited one, which is what makes an
exception possible for abilities and properties alike.
"""

IS_A = "type"
NOT_A = "not_type"
CAN = "can"
CANNOT = "cannot"
HAS_PROPERTY = "property"
LACKS_PROPERTY = "not_property"
HAS_PART = "has"
LACKS_PART = "has_not"

TYPE_RELATIONS = (IS_A, NOT_A)
ABILITY_RELATIONS = (CAN, CANNOT)
PROPERTY_RELATIONS = (HAS_PROPERTY, LACKS_PROPERTY)
PART_RELATIONS = (HAS_PART, LACKS_PART)

# What a second concept is doing in the sentence. Turkish says it with a case
# ending, so the role is read off the word rather than guessed from position.
# How much of a kind a claim covers. Qualitative on purpose: Cyc gave every
# assertion a number from 0 to 100 and abandoned it, because people wrote 95 for
# one and 94 for another with no statistics behind either.
ALL = "hepsi"        # kuşlar uçar        — inherits to every member
MOST = "çoğu"        # çoğu kuş uçar      — a default, but a weak one
SOME = "bazı"        # bazı kuşlar uçmaz  — exists, and blocks the universal
NO = "hiçbiri"       # hiçbir kuş yüzmez  — inherits as a denial

QUANTIFIERS = (ALL, MOST, SOME, NO)
INHERITING = (ALL, NO)      # only these carry down to every member

PLACE = "yer"        # -DA   penguen kutupta yaşar
SOURCE = "çıkış"     # -DAn  kartal serçeden büyüktür
OBJECT = "nesne"     # bare  kediler fare yakalar
