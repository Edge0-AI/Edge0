# edge0 — Framework Python

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

Ce document s'adresse aux **développeurs qui installent et exécutent le
framework Python edge0 depuis les sources**. Il couvre : le démarrage
rapide (installation → modèles → exécution), les performances et la
qualité mesurées, ainsi que la conception de la stack.

edge0 est publié sous forme de **monorepo** —
[`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) — dont le niveau
supérieur contient la documentation partagée (`docs/`) et les
sous-projets par plateforme (`python/` = ce framework, ainsi que
`macos/`, `ios/`, `android/`, `windows/`). Le framework Python est
l'implémentation de référence de la recette edge0 — **SSD expert offload
+ Recover-LoRA + prédiction de routage prerouter** — qui exécute de
grands modèles MoE creux (sparse-MoE) sur Apple Silicon via MLX, avec une
mémoire de pointe limitée par l'ensemble *actif* d'experts plutôt que par
le nombre de paramètres.

Attribution tierce : voir [`NOTICE`](../NOTICE) à la racine du dépôt.
Cartes de modèles : [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
· [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview).
Toutes les commandes ci-dessous s'exécutent depuis ce répertoire
(`python/`).

---

## Démarrage rapide

### Prérequis

| composant | exigence |
|---|---|
| OS / matériel | macOS sur Apple Silicon (M1/M2/M3/M4) — le backend MLX est réservé à Apple Silicon ; un backend CUDA est sur la feuille de route |
| Python | 3.10+ (3.12 recommandé) |
| MLX | `mlx==0.30.6` / `mlx-metal==0.30.6` avec `mlx-lm==0.31.0` (voir `pyproject.toml`) |
| Mémoire | ~2,9 GB actifs de pointe (edge0-35b), ~1,0 GB (edge0-8b), contextes courts |
| Disque | ~23 GB (edge0-35b) / ~4,2 GB (edge0-8b) ; les poids des experts sont mmappés et lus à la demande, pas chargés en RAM au démarrage |

> Une sortie illisible ou mêlant plusieurs langues sur Apple A18 / A18 Pro
> indique une version plus ancienne de `mlx` :
> `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
> ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).

### Installation

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### Modèles

Les deux gammes sont publiées sur Hugging Face et ModelScope ; chaque
dépôt regroupe le checkpoint de base et les adaptateurs LoRA + prerouter
entraînés dans **un seul répertoire**, de sorte qu'un unique
téléchargement constitue un modèle prêt à l'emploi :

```bash
# the repo's helper (defaults to the two published tiers):
.venv/bin/python scripts/fetch_models.py --tier edge0-8b  --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview  --local-dir models/edge0-8b
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview --local-dir models/edge0-35b
```

Pointez edge0 vers une gamme via une variable d'environnement, ou passez
directement le répertoire (la gamme est détectée automatiquement depuis
`config.json`) :

```bash
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

### Exécution

```bash
edge0 demo edge0-8b                 # one-shot generation demo
edge0 chat edge0-8b --prompt "Explain streaming inference in one sentence."
edge0 serve edge0-8b                # OpenAI-compatible server on http://127.0.0.1:8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

`python -m edge0 ...` est équivalent à `edge0 ...`.

### API Python

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier auto-detected
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="Hello!")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # release mmaps / expert cache
```

`examples/demo.py` est le même parcours minimal (`edge0 demo` exécute
exactement ce chemin).

---

## Performances

Mesurées avec `examples/bench.py` (prefill d'un prompt de 3,3 k tokens →
10 pas d'échauffement échantillonnés → 200 tokens de decode
échantillonnés et chronométrés, 2 exécutions par gamme) :

| Gamme | Vitesse de decode | Débit de prefill (à froid / à chaud)* | Mémoire active de pointe | Machine de test |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*À froid = première requête après le démarrage du processus (les poids des
experts arrivent depuis le SSD par défaut de page) ; à chaud = requêtes
suivantes (page cache résident).*

```bash
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
```

## Qualité

Exécutée avec [OpenCompass](https://github.com/open-compass/opencompass)
dans des conditions identiques pour les modèles edge0 (int4 + adaptateurs
entraînés + prerouter) et les bases fp16 d'origine. Le pipeline edge0 perd
peu : **3,9 points en moyenne pour edge0-35b, 2,8 pour edge0-8b**
(MMLU-Pro est même au-dessus de la base). Maximum 100 :

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Moyenne** | **79.2** | **83.2** | **69.9** | **72.7** |

---

## Détails techniques

### Mécanismes fondamentaux

- **SSD expert offload** — les poids des experts sont lus en streaming
  depuis le stockage à la demande ; la mémoire de pointe est limitée par
  l'ensemble actif, et non par le nombre de paramètres.
- **Prerouter** — une tête entraînée prédit le routage des experts un pas
  à l'avance, de sorte que les chargements d'experts chevauchent la passe
  avant au lieu de la bloquer (**jusqu'à +59 %** de débit de decode ; le
  gain augmente avec la latence du stockage, la taille du modèle et la
  largeur routée *K*).
- **Recover-LoRA** — la base int4 est gelée et les adaptateurs LoRA sont
  entraînés par distillation depuis l'enseignant FP, récupérant
  l'essentiel de la perte de quantification 4-bit. Les adaptateurs restent
  non fusionnés : une base unique en lecture seule sert plusieurs jeux
  d'adaptateurs.

### Conception

- **Utilisation façon transformers** — `AutoModel` / `AutoConfig` / `AutoEngine`
  résolvent la gamme à partir du nom du modèle.
- **Isolation du backend** — tout le code MLX se trouve sous
  `src/edge0/backends/mlx/` ; la logique centrale (spécifications des
  modèles, prerouter, pool d'experts en streaming, serveur) ne dépend que
  de la façade du backend (`backends/base.py`), de sorte qu'un nouveau
  backend implémente la même façade (`backends/cuda/` est un emplacement
  réservé) sans aucune modification du code central.
- **Adaptateurs au format safetensors** — les poids LoRA et prerouter sont
  des fichiers `.safetensors` avec métadonnées de provenance (source,
  version, couches propriétaires), résolus depuis le répertoire du modèle
  ou le cache de secours `artifacts/` ignoré par git.
- **Modèle + adaptateurs dans un même répertoire** — un répertoire de
  modèle contient le checkpoint de base et les adaptateurs de ce modèle ;
  la mise à niveau des adaptateurs ne remplace que les fichiers
  d'adaptateurs — la base reste en lecture seule et n'est jamais
  fusionnée.

### Structure du paquet

```
src/edge0/
├── backends/mlx/                  # MLX backend (isolation boundary; cuda/ reserved)
├── engine/  models/  moe/  prerouter/  streaming/
├── adapters/  attention/  server/
└── cli.py  registry.py  sampling.py  config.py
```

---

## Documentation

- [Architecture](../docs/architecture.md)
- [Attention](../docs/attention.md) / [MoE](../docs/moe.md) / [Streaming SSD](../docs/streaming.md) / [prerouter](../docs/prerouter.md)
- [Ajouter un modèle](../docs/adding-a-model.md)
- [edge0-35b](../docs/models/edge0-35b.md) / [edge0-8b](../docs/models/edge0-8b.md)
- Rapport technique : [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](../paper/main.pdf))

## Tests

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q   # real-weight generation
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
```
