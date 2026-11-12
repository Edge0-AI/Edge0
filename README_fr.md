<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**Un framework open source d'inférence MoE en streaming — SSD expert offload + Recover-LoRA + prédiction de routage prerouter.**

**macOS** · **iOS** · **Android** · **Linux** · **Windows** — un seul programme pour tous les modèles Edge0.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--145B-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-145B-q3k)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--120B-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-120B-q3k)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://edge0.ai/discord)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

</div>

## Actualités

- **[2026-11-13]** Nous avons publié **`edge0-145b`** et **`edge0-120b`**, des modèles latent mixture-of-experts dérivés de Qwen3.8-Flash-Next pour un seul GPU NVIDIA sous Linux ou Windows. Ils s'exécutent sur [edge0-strata](edge0-strata/EDGE0.md), que `edge0-infer serve` et `edge0-infer chat` démarrent pour eux ; voir la [page du modèle](docs/models/edge0-145b.md). Poids sur Hugging Face : [`Edge0-145B-q3k`](https://huggingface.co/Edge0/Edge0-145B-q3k) · [`Edge0-145B-NVFP4`](https://huggingface.co/Edge0/Edge0-145B-NVFP4) · [`Edge0-120B-q3k`](https://huggingface.co/Edge0/Edge0-120B-q3k) · [`Edge0-120B-NVFP4`](https://huggingface.co/Edge0/Edge0-120B-NVFP4).
- **[2026-11-04]** Nous avons publié le **moteur d'inférence unifié [`edge0-infer`](edge0-infer/README_fr.md)** : construit sur [GGML](https://github.com/ggml-org/ggml), un seul moteur et une seule API C++ native exécutent le même modèle sur macOS, iOS, Android et Windows, avec une CLI, un serveur local compatible OpenAI et des exemples d'applications de discussion pour le bureau, Android et iOS. La première version prend en charge `edge0-35b` ; `edge0-8b` suivra dans une version ultérieure. Il remplace le framework Python et les moteurs propres à chaque plateforme ; l'ancien code est conservé au tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer).
- **[2026-09-30]** Nous avons publié les **moteurs d'inférence edge0 pour quatre plateformes — iOS, macOS, Android et Windows** (code source au tag [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer)).
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
  [Qualité](#qualité)). Les adaptateurs sont stockés séparément de la base et peuvent être
  activés ou désactivés indépendamment.

### Plateformes

[`edge0-infer`](edge0-infer/README_fr.md) est le programme que vous lancez (CLI, serveur local compatible OpenAI et Anthropic, applications de discussion), avec les mêmes commandes pour tous les modèles. Chaque modèle s'exécute sur le moteur conçu pour lui, et edge0-infer choisit le moteur d'après le dossier du modèle ; il n'y a rien à choisir : `edge0-35b` s'exécute sur le **moteur natif** de `edge0-infer/` (construit sur GGML, avec une API C++ native), et `edge0-145b` et `edge0-120b` sur **edge0-strata**, dans [`edge0-strata/`](edge0-strata/EDGE0.md), un fork de Strata.

| Plateforme | `edge0-35b` (moteur natif) | `edge0-145b`, `edge0-120b` (edge0-strata) |
|---|---|---|
| **macOS** (Apple Silicon) | Metal, CPU | — |
| **iOS** | Metal | — |
| **Android** | CPU | — |
| **Linux** | CPU | GPU NVIDIA (CUDA 13) |
| **Windows** | CPU, Vulkan | GPU NVIDIA (CUDA 13) |

Mesures : [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures) pour `edge0-35b`, et la [page du modèle](docs/models/edge0-145b.md) pour `edge0-145b` et `edge0-120b`. Sur le moteur natif, le prefill et le decode peuvent s'exécuter sur des backends différents du même appareil, par exemple prefill sur Metal et decode sur CPU. Les exemples d'applications de discussion pour le bureau (macOS, Windows), Android et iOS se trouvent dans [`edge0-infer/examples/`](edge0-infer/examples/README.md#chat-applications).

### Modèles

Deux gammes de modèles, `edge0-35b` et `edge0-8b`, sont fournies avec le framework. Chaque gamme est
une publication complète : le checkpoint publié, les adaptateurs LoRA
entraînés et les têtes prerouter entraînées fonctionnent ensemble comme
une seule unité.

| Gamme | Checkpoint publié | Profil d'inférence | Moteur |
|---|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4-bit, 40 couches, 256 experts, prerouter K=4 | Moteur natif ✅ |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4-bit, 24 couches, 128 experts, prerouter K=8 | Moteur natif (prévu) |
| [`edge0-145b`](docs/models/edge0-145b.md) | [`Edge0/Edge0-145B-q3k`](https://huggingface.co/Edge0/Edge0-145B-q3k) · [NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) | experts latents Q3_K ou NVFP4 (1792), 48 couches, 512 experts, K=10 | edge0-strata ✅ (Linux ou Windows, GPU NVIDIA) |
| [`edge0-120b`](docs/models/edge0-145b.md) | [`Edge0/Edge0-120B-q3k`](https://huggingface.co/Edge0/Edge0-120B-q3k) · [NVFP4](https://huggingface.co/Edge0/Edge0-120B-NVFP4) | experts latents Q3_K ou NVFP4 (1280), 48 couches, 512 experts, K=10 | edge0-strata ✅ (Linux ou Windows, GPU NVIDIA) |

Les deux checkpoints publiés reposent sur des modèles de base sparse-MoE ouverts
(respectivement Qwen3.6-35B-A3B et l'hybride bailing Ling 3.0) et intègrent
l'entraînement LoRA et prerouter réalisé pour ce framework. edge0-infer
télécharge le checkpoint publié avec ses outils de ressources et le convertit
en ressources de modèle du moteur (voir [Démarrage rapide](#démarrage-rapide)).
`edge0-145b` et `edge0-120b` forment une gamme latent-MoE distincte pour un seul GPU NVIDIA ; ils s'exécutent avec edge0-strata, que `edge0-infer serve` et `edge0-infer chat` démarrent (voir [edge0-145b et edge0-120b](docs/models/edge0-145b.md)).

### Conception

- **Basé sur GGML, le même code sur chaque plateforme** : la bibliothèque
  native ne dépend que de GGML, intégré sans modification dans une version
  fixe ; la compilation ne télécharge rien ;
- **Une seule API native** : les applications se lient à la cible CMake
  `edge0::edge0` et incluent les en-têtes de `include/edge0/` pour utiliser le
  moteur dans le processus ; la CLI de référence et le serveur local
  utilisent les mêmes paramètres et valeurs par défaut que l'API
  native ;
- **Experts à la demande** : les experts routés sont lus depuis le stockage à
  la demande dans un nombre fixe d'emplacements par couche, si bien qu'un
  modèle plus grand que le budget mémoire de l'appareil s'exécute sans être
  chargé en entier ;
- **Deux modes de routage** : `predicted` (par défaut) utilise le prerouter
  pour préparer en arrière-plan les experts du token suivant pendant le calcul
  du token courant ; `actual` est le calcul d'origine du modèle, pour
  comparaison ;
- **Réutilisation multi-tours** : dans une conversation multi-tours, seul le
  nouveau message de chaque tour est généralement calculé ; les résultats
  précédents sont réutilisés.

Voir l'[architecture d'edge0-infer](edge0-infer/docs/architecture.md).

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

`edge0-145b` et `edge0-120b` (**gras** = meilleur de la colonne ; voir la [page du modèle](docs/models/edge0-145b.md#quality)) :

| Model | deepswe 1.1 | terminal-bench 2.1 | AA-LCR | IFBench | GPQA-Diamond | HLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge0-145B** | **64.6** | **86.5** | 79.3 | 80.6 | 90.4 | 35.4 |
| **Edge0-120B** | 51.3 | 75.3 | 81.0 | 79.3 | 87.4 | 31.0 |
| DeepSeek-V4-Flash-0731 | 54.4 | 82.7 | 79.7 | 79.2 | 90.8 | 33.8 |
| Qwen3.8-Flash-Next | 58.7 | 83.3 | 79.1 | **81.3** | **91.7** | 35.9 |
| Qwen3.8-27B | 42.2 | 73.0 | **82.0** | 79.5 | 89.2 | 30.8 |
| Claude-Opus-4.6 (Max) | / | 78.2 | 78.0 | 62.5 | 91.3 | **40.0** |

### Benchmark

`edge0-35b` avec le routage `predicted`, mesuré avec edge0-infer (chaque plage va du minimum au maximum de trois processus) :

| Plateforme | Appareil | Prompt | Premier token | Vitesse de decode | Pic mémoire |
|---|---|---|---|---|---|
| macOS Metal | Apple M5, 16 Gio | 3 084 tokens | 5,60–5,87 s | 24,8–24,9 tok/s | 3,43 Gio (`phys_footprint`) |
| macOS CPU | Apple M5, 16 Gio | 627 tokens | 11,5–11,7 s | 21,7–22,3 tok/s | 2,96 Gio (`phys_footprint`) |
| iOS Metal | iPhone 17 Pro Max, 12 Go | 627 tokens | 8,20–8,26 s | 13,2 tok/s | 3,05 Gio (`phys_footprint`) |
| Android CPU | OnePlus PKR110, 16 Go | 627 tokens | 18,4–18,8 s | 11,4–11,8 tok/s | 2,82 Gio (RSS) |
| Windows CPU | Intel Core i5-12400, 32 Go | 627 tokens | 16,9–17,3 s | 7,3–8,2 tok/s | 2,86 Gio (jeu de travail) |
| Windows Vulkan (deux phases, 32 emplacements d'experts) | RTX 4070 SUPER, 12 Go | 627 tokens | 3,75–3,76 s | 23,6–24,0 tok/s | Hausse de la mémoire GPU 5,35 Gio |

Les métriques mémoire diffèrent selon la plateforme et ne sont pas directement comparables. La méthode de mesure, les chiffres en routage `actual` et les commandes de reproduction se trouvent dans [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures).

`edge0-145b` et `edge0-120b` sur edge0-strata, un seul GPU NVIDIA (tok/s) :

#### edge0-145b

| Machine | Build | 4K prefill | 16K prefill | Chat decode | Agent decode |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 D 24 GB | Q3_K | 2169.9 | 3554.9 | 99.7 | **121.2** |
| RTX 5090 D 24 GB | NVFP4 | 1839.6 | 3165.7 | 79.6 | 98.3 |
| RTX 4090 24 GB | Q3_K | 2042.0 | 3179.4 | 69.9 | 91.7 |
| RTX Spark N1X 60 GiB | Q3_K | 674.0 | 770.3 | 37.7 | 36.7 |
| RTX Spark N1X 60 GiB | NVFP4 | 821.2 | 922.4 | 40.1 | 46.0 |
| 16 GiB VRAM + 20 GiB RAM | Q3_K | 249 | 335 | 36.8 | 44.9 |

#### edge0-120b

| Machine | Build | 4K prefill | 16K prefill | Chat decode | Agent decode |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 D 24 GB | Q3_K | 3236.7 | 4576.5 | 130.2 | **144.6** |
| RTX 5090 D 24 GB | NVFP4 | 2668.5 | 4357.1 | 124.9 | 135.5 |
| RTX 4090 24 GB | Q3_K | 2637.5 | 3551.1 | 95.0 | 104.9 |
| RTX Spark N1X 60 GiB | Q3_K | 987.0 | 1081.7 | 47.2 | 68.4 |
| RTX Spark N1X 60 GiB | NVFP4 | 996.4 | 1066.9 | 45.1 | 68.9 |
| 16 GiB VRAM + 20 GiB RAM | Q3_K | 430 | 778 | 62.4 | 73.3 |
| 16 GiB VRAM + 20 GiB RAM | NVFP4 | 361.1 | 619.7 | 51.3 | 59.9 |

Les lignes `16 GiB VRAM + 20 GiB RAM` correspondent à la même RTX 5090 D 24 GB, limitée par cgroup à 16 Gio de mémoire GPU et 20 Gio de mémoire système. Les détails sont sur la [page du modèle](docs/models/edge0-145b.md#speed-measured-one-card).

## Démarrage rapide

Lancez les commandes suivantes dans `edge0-infer/`. Il faut CMake 3.21 ou
plus récent, Ninja, un compilateur C++17/C11, uv et Git ; le 35B nécessite
environ 18 Go pour les poids d'origine et environ 20 Go pour les ressources
converties, plus autant d'espace libre pendant la conversion.

### 1) Compiler

```bash
cd edge0-infer
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # doit afficher une ligne : model.0: edge0_35b
```

Sur les machines sans Metal, utilisez `--preset cpu`, `build/cpu/bin/edge0-infer`
et `--device cpu` dans les commandes ci-dessous.

### 2) Obtenir un modèle

**edge0-35b (moteur natif).** Téléchargez et convertissez les ressources du modèle :

```bash
export MOE_SOURCE_35B=<répertoire des poids d'origine>
export MOE_ASSETS_35B=<répertoire des ressources converties>

uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }

assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
```

Le téléchargement récupère `Edge0/Edge0-35B-A3B-preview` sur Hugging Face et le
vérifie avec la révision et les empreintes verrouillées.

**edge0-145b et edge0-120b (edge0-strata).** Compilez edge0-strata comme l'indique [`edge0-strata/EDGE0.md`](edge0-strata/EDGE0.md) et utilisez un dossier de modèle publié (voir [edge0-145b et edge0-120b](docs/models/edge0-145b.md)).

### 3) Exécuter

```bash
# une requête
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Présente-toi en une phrase."
# discussion multi-tours
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
# serveur (compatible OpenAI et Anthropic, par défaut http://127.0.0.1:8000)
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

Pour edge0-145b et edge0-120b, `chat` et `serve` prennent le dossier du modèle de la même façon (`--model <dossier du modèle>`) ; `run` est réservé au moteur natif.

Pour les compilations Android, Windows et iOS, et pour utiliser le moteur depuis
votre propre projet CMake, voir [Getting started](edge0-infer/docs/getting-started.md).

## Documentation

**Modèles**

- [Routage MoE et experts](docs/moe.md) / [prerouter](docs/prerouter.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md) / [edge0-145b et edge0-120b](docs/models/edge0-145b.md)
- Rapport technique : [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

**edge0-infer**

- [Getting started](edge0-infer/docs/getting-started.md) : compilation, ressources du modèle, exécution, discussion et serveur sur chaque plateforme
- [Configuration](edge0-infer/docs/configuration.md) / [Native API](edge0-infer/docs/native-api.md) / [Serveur local](edge0-infer/docs/local-server.md)
- [Architecture](edge0-infer/docs/architecture.md) / [Known limitations](edge0-infer/docs/limitations.md)
- [Benchmarking](edge0-infer/docs/benchmarking.md) / [Testing](edge0-infer/docs/testing.md)
- [Outils de ressources du modèle et ajout d'un modèle](edge0-infer/tools/assets/README.md)

**Moteurs**

- Moteur natif : [Architecture](edge0-infer/docs/architecture.md#native-engine) / [Native API](edge0-infer/docs/native-api.md)
- edge0-strata : [compilation et contrat de lancement](edge0-strata/EDGE0.md) / [edge0-145b](edge0-infer/docs/edge0-145b.md) et [edge0-120b](edge0-infer/docs/edge0-120b.md) dans edge0-infer

## Feuille de route

### T4 2026

**Plateformes et systèmes**

- ✅ **Moteur d'inférence unifié edge0 publié en open source** (2026-11-04) —
  [`edge0-infer`](edge0-infer/README_fr.md) : un seul moteur et une seule API
  native pour macOS, iOS, Android et Windows.
- **Prise en charge d'`edge0-8b` dans edge0-infer**.

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

La compilation et les tests sont décrits dans [Testing](edge0-infer/docs/testing.md) ;
les vérifications qui ne nécessitent pas de ressources de modèle :

```bash
cd edge0-infer
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"
```

Pour les modifications sous `edge0-infer/`, ouvrez comme d'habitude des
issues et des PR dans ce dépôt ; le répertoire est développé dans un dépôt
séparé : les mainteneurs appliquent d'abord là-bas les modifications
acceptées, qui arrivent ici lors de la synchronisation suivante. Voir
[CONTRIBUTING](CONTRIBUTING.md).

Workflow : fork → branche de fonctionnalité → PR vers `main`. Merci de
garder les tests au vert et d'ajouter des tests pour les nouveaux
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

- **Discord** : [edge0.ai/discord](https://edge0.ai/discord) — discussion et support communautaires
- **Email** : samuel@edge0.ai

Pour les bugs et les demandes de fonctionnalités, merci d'utiliser
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## Licence

Apache-2.0 ; les composants tiers intégrés conservent leur propre licence (voir [NOTICE](NOTICE)).
