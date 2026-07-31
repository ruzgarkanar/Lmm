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

TYPE_RELATIONS = (IS_A, NOT_A)
ABILITY_RELATIONS = (CAN, CANNOT)
PROPERTY_RELATIONS = (HAS_PROPERTY, LACKS_PROPERTY)
