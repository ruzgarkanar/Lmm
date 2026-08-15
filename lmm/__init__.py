"""LMM katmanı: Qwen (dil) + graf/kapı (doğruluk & büyüme).

Belkemiği kural — `v3/gate.py`'nin "geometri kayıt yazamaz" kuralının aynısı:
QWEN KAYIT YAZAMAZ. Qwen yalnız ADAY üretir (metin, çıkarım, cevap); belleğe
neyin gireceğine ve neyin söyleneceğine KAPI karar verir. Bu tek cümle tüm
modülün belkemiğidir.

Yeniden kullanılan çekirdek (değişmeden): v3/memory.py · v3/gate.py ·
v3/geometry.py · v3/dynamics.py.
"""
