<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**Un framework open source d'inférence MoE en streaming — SSD expert offload + Recover-LoRA + prédiction de routage prerouter.**

**Python** · **macOS** · **iOS** · **Android** — une seule recette, tous les appareils.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

</div>

## Actualités

- **[2026-09-30]** Nous avons publié les **moteurs d'inférence edge0 pour quatre plateformes — iOS, macOS, Android et Windows** — afin d'offrir aux utilisateurs la meilleure expérience d'inférence à travers les architectures et les plateformes. Le code source est publié en open source dans ce dépôt ([`ios/`](ios/) · [`macos/`](macos/) · [`android/`](android/) · [`windows/`](windows/)) — consultez le README de chaque répertoire pour plus de détails. Le **framework d'inférence unifié** suivra au **T4 2026** ; voir la [Feuille de route](#feuille-de-route).
- **[2026-09-16]** Notre rapport technique est sur arXiv : [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063).
- **[2026-09-08]** Première publication open source d'**edge0**, avec les deux gammes de modèles — [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) et [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) — sur Hugging Face et ModelScope.

## À propos

**edge0** est un framework open source d'inférence MoE en streaming. Il
généralise la recette éprouvée en production — **SSD expert offload +
Recover-LoRA + prédiction de routage prerouter** — en un framework
extensible qui exécute de grands modèles MoE creux (sparse-MoE) sur du
matériel grand public : la mémoire de pointe est limitée par l'ensemble
*actif* d'experts, et non par le nombre de paramètres.

### Mécanismes fondamentaux

- **SSD expert offload** : les poids des experts sont lus en streaming
  depuis le stockage à la demande ; la mémoire de pointe est limitée par
  l'ensemble actif, et non par le nombre de paramètres.
- **Prerouter** : une tête entraînée prédit le routage des experts un pas
  à l'avance, de sorte que les chargements d'experts chevauchent la passe
  avant au lieu de la bloquer — **jusqu'à +59 %** de débit de decode ; le
  gain augmente avec la latence du stockage, la taille du modèle et la
  largeur routée *K*.
- **Recover-LoRA** : la base int4 est gelée et les adaptateurs LoRA sont
  entraînés par distillation depuis l'enseignant FP, récupérant
  l'essentiel de la perte de quantification à 4 bits (voir
  [Qualité](#qualité)). Les adaptateurs restent non fusionnés : une base
  unique en lecture seule sert plusieurs jeux d'adaptateurs.

### Plateformes

Un dépôt, une recette, des runtimes par plateforme :

| Plateforme | Répertoire | Stack | Statut |
|---|---|---|---|
| **Python** (macOS · Apple Silicon) | [`python/`](python/README_fr.md) | Python + MLX | ✅ Disponible maintenant |
| Application et CLI **macOS** | [`macos/`](macos/README_fr.md) | Rust | ✅ Open source (2026-09-30) |
| Application **iOS** | [`ios/`](ios/README_fr.md) | Swift + MLX Swift | ✅ Open source (2026-09-30) |
| Application et moteur **Android** | [`android/`](android/README_fr.md) | Kotlin + moteur natif | ✅ Open source (2026-09-30) |
| Application et moteur **Windows** | [`windows/`](windows/README_fr.md) | C++ + HIP (ROCm) / Vulkan | ✅ Open source (2026-09-30) |

### Modèles

Deux gammes de modèles sont fournies avec le framework. Chaque gamme est
une publication complète : le checkpoint publié, les adaptateurs LoRA
entraînés et les têtes prerouter entraînées fonctionnent ensemble comme
une seule unité.

| Gamme | Checkpoint publié | Profil d'inférence |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit, 40 couches, 256 experts, prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit, 24 couches, 128 experts, prerouter K=8 |

Les deux checkpoints sont construits à partir de modèles de base MoE creux
open source (respectivement Qwen3.6-35B-A3B et l'hybride bailing Ling 3.0)
et sont fournis avec l'entraînement LoRA et prerouter réalisé pour ce
framework — les fichiers d'adaptateurs sont colocalisés avec chaque
checkpoint et se chargent automatiquement, de sorte que
`edge0 serve <tier>` exécute le pipeline entraîné prêt à l'emploi.

### Conception

- **Utilisation façon transformers** : `AutoModel` / `AutoConfig` / `AutoEngine`
  résolvent la gamme à partir du nom du modèle ;
- **Isolation du backend (framework Python)** : dans le framework Python,
  tout le code MLX se trouve sous `python/src/edge0/backends/mlx/` ; la
  logique centrale (spécifications des modèles, prerouter, pool d'experts
  en streaming, serveur) ne dépend que de la façade du backend
  (`backends/base.py`), de sorte qu'un nouveau backend implémente la même
  façade (`backends/cuda/` est un emplacement réservé) sans aucune
  modification du code central. Les moteurs iOS / macOS / Android sont
  aujourd'hui fournis avec des stacks natives par plateforme — regrouper
  toutes les plateformes sous une couche d'accès unique est précisément ce
  que fournira le framework d'inférence unifié (voir la
  [Feuille de route](#feuille-de-route)) ;
- **Adaptateurs au format safetensors** : les poids LoRA et prerouter sont
  des fichiers `.safetensors` avec métadonnées de provenance (source,
  version, couches propriétaires), résolus depuis le répertoire du modèle
  ou `artifacts/` ;
- **Modèle + adaptateurs dans un même répertoire** : un répertoire de
  modèle contient à la fois le checkpoint de base (`config.json` /
  `model*.safetensors` / tokenizer) et les adaptateurs de ce modèle ; la
  mise à niveau des adaptateurs ne remplace que les fichiers d'adaptateurs
  — la base reste en lecture seule et n'est jamais fusionnée.

### Qualité

Tous les benchmarks ont été exécutés par nos soins avec
[OpenCompass](https://github.com/open-compass/opencompass), dans des
conditions et avec des paramètres identiques, pour les modèles edge0
(int4 + adaptateurs entraînés + routage prerouter) et pour les modèles de
base fp16 d'origine. La perte du pipeline edge0 est faible : **3,9 points
en moyenne pour edge0-35b, 2,8 pour edge0-8b** (MMLU-Pro est même
au-dessus de la base). Maximum 100 :

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Moyenne** | **79.2** | **83.2** | **69.9** | **72.7** |

### Benchmark

Mesuré avec `python/examples/bench.py` (prefill d'un prompt de 3,3 k
tokens → 10 pas d'échauffement échantillonnés → 200 tokens de decode
échantillonnés et chronométrés, 2 exécutions par gamme) :

| Gamme | Vitesse de decode | Débit de prefill (à froid / à chaud)* | Mémoire active de pointe | Machine de test |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*À froid = première requête après le démarrage du processus (les poids des
experts arrivent depuis le SSD par défaut de page) ; à chaud = requêtes
suivantes (page cache résident). Les chiffres de prefill sont des débits
mesurés sur un prompt d'environ 3,3 k tokens (`BENCH_LONG=1`).*

Reproduire :

```bash
cd python
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
```

## Démarrage rapide

### Python (macOS · Apple Silicon)

#### Prérequis

- **OS / matériel** : le backend MLX fonctionne sur macOS avec Apple
  Silicon (M1/M2/M3/M4). Le backend CUDA est sur la feuille de route —
  aucune autre plateforme n'est encore prise en charge par le framework
  Python.
- **Python** : 3.10+ (3.12 recommandé).
- **MLX** : `mlx==0.30.6` / `mlx-metal==0.30.6` avec `mlx-lm==0.31.0` (voir
  `python/pyproject.toml`). Une sortie illisible ou mêlant plusieurs
  langues sur Apple A18 / A18 Pro indique une version plus ancienne de
  `mlx` : `pip install 'mlx==0.30.6'
  'mlx-metal==0.30.6'` ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).
- **Mémoire** : environ 2,9 GB de mémoire active de pointe pour
  `edge0-35b`, environ 1,0 GB pour `edge0-8b` (contextes courts ; voir
  [Benchmark](#benchmark)). Prévoyez de la marge pour l'OS, le tokenizer
  et la croissance de la KV cache en contexte long.
- **Disque** : les checkpoints 4-bit pèsent environ 23 GB (`edge0-35b`) et
  environ 4,2 GB (`edge0-8b`) ; les poids des experts sont mmappés et lus
  à la demande, ils ne sont pas chargés en RAM au démarrage.

#### 1) Installer

```bash
cd python
# Python >= 3.10; the MLX backend requires macOS with Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

#### 2) Télécharger un modèle

Les deux gammes sont publiées sur Hugging Face et ModelScope — chaque
dépôt regroupe le checkpoint de base et les adaptateurs LoRA + prerouter
entraînés dans **un seul répertoire**, de sorte qu'un unique
téléchargement constitue un modèle prêt à l'emploi :

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) (~23 GB) · [Miroir ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) (~4,2 GB) · [Miroir ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)

```bash
# with the repo's helper (defaults to the two repos above):
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview     --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview     --local-dir models/edge0-8b
```

Quelle que soit la méthode, vous obtenez un répertoire comme celui-ci :

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer files   # base checkpoint
├── lora_edge0_35b.safetensors          # trained LoRA adapters
└── prerouter_edge0_35b.safetensors     # trained prerouter heads
```

#### 3) Pointer edge0 vers le modèle

Les noms de gamme sont résolus vers des répertoires locaux via des
variables d'environnement (l'emplacement du téléchargement est libre) :

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

Ou ignorez complètement les variables d'environnement et passez
directement le répertoire — la gamme est détectée automatiquement depuis
le `config.json` du checkpoint :

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

#### 4) Exécuter

```bash
# quick demo
edge0 demo edge0-35b

# serve (OpenAI-compatible /v1/chat/completions)
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"edge0-35b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) One-shot chat (pass --max-new to cap length; add --show-thinking to
#    print the model's reasoning block too)
edge0 chat edge0-35b --prompt "Explain streaming inference in one sentence."
```

`python -m edge0 ...` est équivalent à `edge0 ...`.

#### API Python

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

`python/examples/demo.py` est le même parcours minimal (`edge0 demo`
exécute exactement ce chemin).

#### Modèles et adaptateurs

- **Checkpoint** : le répertoire du modèle d'origine (`config.json`,
  `model*.safetensors`, tokenizer). `edge0 serve <dir>` /
  `AutoEngine.from_pretrained(<dir>)` détectent la gamme depuis
  `config.json`.
- Les **adaptateurs** (LoRA + prerouter, safetensors) sont résolus
  automatiquement depuis l'un ou l'autre emplacement :
  - le répertoire du modèle (recommandé) : à côté de la base, par exemple
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors` ;
  - `artifacts/` à la racine du projet Python (ignoré par git) : un cache
    de secours optionnel pour les safetensors d'adaptateurs non
    colocalisés avec le modèle.
- Les dépôts de modèles publiés regroupent à la fois le checkpoint de base
  et la version par défaut actuelle des adaptateurs, de sorte que
  `scripts/fetch_models.py` produit un répertoire de modèle prêt à
  l'emploi. Consultez la page de documentation de chaque modèle pour la
  provenance de ses adaptateurs (données d'entraînement, disposition des
  couches propriétaires).
- Les deux adaptateurs sont requis pour le pipeline prerouter + LoRA ; si
  un fichier manque, `edge0` échoue avec un message clair (ou passez
  `--no-prerouter` / `--no-lora` pour exécuter le modèle de base seul).

#### Documentation

- [Architecture](docs/architecture.md)
- [Attention](docs/attention.md) / [MoE](docs/moe.md) / [Streaming SSD](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [Ajouter un modèle](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- Rapport technique : [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

### macOS / iOS / Android / Windows

Les quatre moteurs par plateforme sont publiés en open source dans ce
dépôt — plus de détails dans le README de chaque répertoire :

- **macOS** : CLI locale / démon / application de bureau (Rust) — voir [`macos/README_fr.md`](macos/README_fr.md)
- **iOS** : application iPhone embarquée (Swift + MLX Swift) — voir [`ios/README_fr.md`](ios/README_fr.md)
- **Android** : application embarquée + moteur natif (Kotlin) — voir [`android/README_fr.md`](android/README_fr.md)
- **Windows** : application de bureau + moteur natif (C++ + HIP (ROCm) / Vulkan) — voir [`windows/README_fr.md`](windows/README_fr.md)

Le **framework d'inférence unifié** — une couche d'accès unique, un
runtime s'adaptant automatiquement à iOS / macOS / Android / Windows /
Python — arrivera au **T4 2026** ; voir la
[Feuille de route](#feuille-de-route).

## Feuille de route

### T4 2026

**Plateformes et systèmes**

- **Framework d'inférence unifié edge0** — nous publierons en open source
  un framework d'inférence unifié : **une couche d'accès unifiée** (une
  API unique pour le chat / le serving / l'utilisation embarquée), avec le
  **runtime s'adaptant automatiquement à la plateforme matérielle** — iOS,
  macOS, Android, Windows et Python. Il s'appuie sur les moteurs par
  plateforme déjà publiés en open source dans ce dépôt (`ios/` · `macos/`
  · `android/` · `windows/`).
- **Backend CUDA** pour le framework Python — emplacement réservé dans
  `python/src/edge0/backends/cuda/`, le code central ne nécessite aucune
  modification.

**Modèles et algorithmes**

Le T4 travaille sur deux fronts : intégrer une architecture de nouvelle
génération dans le framework, et transformer le raisonnement latent en un
véritable gain de latence et non seulement arithmétique.

- **Prise en charge des architectures de nouvelle génération (classe
  Qwen3.8-Flash)** — exécuter sur edge0 l'attention linéaire hybride
  (GDN + QSA), les résidus multi-branches à portail et les topologies
  d'embedding N-gram. Ces conceptions se prêtent naturellement au SSD
  expert offload en streaming : l'attention à état O(1) évite que les
  longues réflexions ne deviennent un problème de KV cache, et les tables
  N-gram, consultables uniquement en lecture (lookup-only), se chargent en
  streaming à la demande. Objectif : que la gamme s'exécute sur un seul
  appareil et que les benchmarks se situent à un écart acceptable de la
  base fp16.
- **Raisonnement latent + pré-prédiction d'experts par lots** — faire du
  raisonnement latent une économie de *latence*, et pas seulement
  arithmétique. Le problème d'ingénierie central : faire passer le routage
  des experts d'une prédiction par position à **une prédiction par bloc**,
  afin qu'une seule prédiction couvre toutes les positions et tous les
  tours d'un bloc et que **le volume de chargement des experts soit
  découplé du nombre de tours de la boucle de raisonnement** — plus un
  préchargement inter-blocs qui charge les experts du bloc suivant dans la
  fenêtre de calcul du bloc courant. Les progrès sont mesurés en **temps
  de phase de réflexion de bout en bout à précision égale** (jamais en
  tokens/s).
- Davantage de gammes de modèles et de versions d'adaptateurs sur le
  pipeline existant.

## Contribution

Les contributions sont les bienvenues — issues, PR, rapports de benchmark
et portages de modèles, tout compte.

**Framework Python** (disponible maintenant) :

```bash
cd python
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'

pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                       # real-weight generation; missing tiers are skipped
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                       # staged vs exact consistency + generation smoke
scripts/generate_example.py   # full-pipeline API example
examples/demo.py              # minimal API walkthrough
```

La CI exécute les tests unitaires (macOS + MLX) et une suite d'hygiène du
dépôt (pas de chemins codés en dur, vérification des frontières de backend
et des secrets) sur chaque PR.

**Runtimes par plateforme** (macOS / iOS / Android / Windows) : chaque
répertoire de plateforme dispose de son propre guide de compilation et de
ses tests — voir le README de chaque répertoire.

Workflow : fork → branche de fonctionnalité → PR vers `main`. Merci de
garder la suite d'hygiène au vert et d'ajouter des tests pour les nouveaux
comportements.

## Citation

Si vous trouvez edge0 utile, merci de citer notre rapport technique :

```bibtex
@article{lin2026other,
  title   = {The Other Half of the Memory Wall: Serving 35B MoEs from SSD
             with Trained Routing Prediction},
  author  = {Lin, Yu and Wang, Yiming and Cai, Runyuan and Liu, Hanze and
             Zeng, Xiaodong},
  journal = {arXiv preprint arXiv:2609.18063},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.18063}
}
```

## Nous contacter

Les canaux communautaires et d'assistance arrivent bientôt — cette section
listera les moyens officiels de nous joindre :

- **Email** : samuel@edge0.ai

Pour les bugs et les demandes de fonctionnalités, merci d'utiliser
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## Licence

Apache-2.0, y compris le code tiers intégré (voir [NOTICE](NOTICE)).
