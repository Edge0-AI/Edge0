"""guard.py - il contatore delle richieste del server, per accorgersi che un altro client ha parlato con la
porta 8080 durante una misura.

Il server tiene un totale cumulativo delle richieste finite (serve/server.py: `self.totals = {... "requests": 0 ...}`,
incrementato a richiesta completata) e lo pubblica in GET /metrics sotto "totals". Presente in tutte le build in
C:\\strata-lab\\pr dalla 0.1.26 in poi; l'endpoint non chiede chiave quando il config non ne ha una.

Uso: leggere il contatore subito prima e subito dopo le proprie richieste e controllare che la differenza sia
esattamente il numero di richieste fatte dallo script. Se e' maggiore, qualcun altro ha generato su quel server
(Jan, una scheda del browser aperta sulla chat, un altro script): lo stato del checkpoint fra una richiesta e
l'altra non e' piu' quello che credi e la misura va buttata, non interpretata.
"""
from __future__ import annotations

import json
import urllib.request


def requests_total(url: str, timeout: float = 30) -> int:
    """Il contatore cumulativo delle richieste finite, da GET /metrics."""
    req = urllib.request.Request(url.rstrip("/") + "/metrics")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["totals"]["requests"]


def report_extra(extra: int, what: str, prima: int = -1, dopo: int = -1) -> bool:
    """Stampa l'esito del controllo. Ritorna True se la misura e' pulita.

    Stampa anche i due valori assoluti del contatore: servono perche' il controllo del delta vede solo la propria
    finestra. Una richiesta estranea arrivata PRIMA della misura (fra l'avvio del server e lo script, o fra due
    script della stessa sequenza) sporca lo stato lo stesso e qui non si vede: su un server appena avviato il
    contatore prima della sonda deve essere 0, e prima di gen_det deve essere il numero di richieste che la
    sequenza ha gia' fatto (nel cancello: 1, la sonda).
    """
    print(f"contatore del server: prima {prima}, dopo {dopo}")
    if extra == 0:
        print(f"controllo porta: nessuna richiesta estranea durante {what}")
        return True
    quante = "1 richiesta non mia" if extra == 1 else f"{extra} richieste non mie"
    print(f"*** MISURA SPORCA: {quante} durante {what}.")
    print("*** Un altro client ha generato su questa porta: lo stato fra una richiesta e l'altra e' cambiato.")
    print("*** Chiudi Jan e le schede del browser sulla chat, riavvia il server e rifai la misura con un'altra etichetta.")
    return False
