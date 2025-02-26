Implementacija hierarhičnega razvrščanja
========================================

V sklopu domače naloge boste implementirali hierarhično razvrščanje v skupine.

Predlogo `hc.py` boste morali dopolniti z implementacijami `euclidean_dist`, `manhattan_dist`, `average_linkage`, `single_linkage`, `complete_linkage`,  `HierarchicalClustering.closest_clusters` in `HierarchicalClustering.run`. 

Vhod in izhod posameznih funkcij definirajo testi `test_hc.py`, pri razumevanju bodo pa pomagali tudi komentarji v predlogi in primer uporabe na koncu datoteke.

Predlagamo naslednji vrsti red reševanja:

1. Začnite z osnovno implementacijo, torej tako, ki gradi le strukturo skupin (brez razdalj) in ki ne rešuje problema neznanih vrednosti (testni razred `HierarchicalClusteringTest`).

2. Rešitev nadgradite z ustreznim obravnavanjem neznanih vrednosti (testni razred `HierarchicalClusteringUnknownsTest`). Pri primerjavi posameznih vektorjev pare števil, kjer je kakšna vrednost neznana, izpustite. Ker slednje strogo zmanjša razdalje, jih je treba normalizirati na takšne, kot bi jih z isto povprečno razliko po elementih pričakovali ob originalni dolžini vektorjev. Pri računanju razdalj med skupinami pare z neznanimi razdaljami izpustite. Pri izboru najbližjih skupin podobno izpustite pare skupin, ki jih ne morete poračunati. 

3. Rešitev nadgradite tako, da lahko opcijsko vrača razdalje za vse združene pare skupin (testni razred `HierarchicalClusteringWithDistancesTest`). Takšna struktura vsebuje vse potrebne informacije za izris dendrograma.

Rešitev mora delovati s Pythonom 3.12 in prestati teste v `test_hc.py`. Uporaba modulov, ki niso v [standardni knjižnici](https://docs.python.org/3/library/), kot je recimo `numpy`, pri tej nalogi še ni dovoljena. Ker ne potrebujete posebnih knjižnic, tudi okolja ne bo težko vzpostaviti.

Teste lahko poženete z ukazom `python test_hc.py`, skoraj gotovo pa vaš izbran urejevalnik nudi kaj lepšega. Za namige se obrnite na nas.