# edge0-infer

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

`edge0-infer` exécute des modèles de langage à mélange d'experts (MoE) sur votre propre matériel : une CLI, un serveur local compatible OpenAI et des applications de discussion, avec les mêmes commandes pour tous les modèles. Chaque modèle s'exécute sur le moteur conçu pour lui, et `edge0-infer` choisit le moteur d'après le dossier du modèle ; il n'y a rien à choisir :

- **Edge0 35B** (Qwen3.6-35B-A3B avec un LoRA et un pre-router entraîné) s'exécute sur le **moteur natif** de ce répertoire, construit sur [GGML](https://github.com/ggml-org/ggml) et doté d'une API C++ native, sous macOS, iOS, Android, Linux et Windows. Les poids des experts sont chargés depuis le stockage à la demande dans un ensemble fixe d'emplacements, si bien qu'un modèle plus grand que le budget mémoire de l'appareil s'exécute sans charger tous ses poids. Sur un Mac Apple M5, il décode à environ 25 tokens par seconde, avec un pic d'empreinte physique d'environ 3,4 Gio ; les chiffres de chaque plateforme se trouvent dans [Benchmarking](docs/benchmarking.md#reference-figures).
- **Edge0 145B** et **Edge0 120B** s'exécutent sur **edge0-strata**, un fork de [Strata](https://github.com/Niko1221/Strata) dans `edge0-strata/` du dépôt Edge0, sur un GPU NVIDIA sous Linux ou Windows ; voir [Edge0 145B](docs/edge0-145b.md) et [Edge0 120B](docs/edge0-120b.md).

> Cette page est une traduction du [README](README.md) en anglais ; les documents liés sont en anglais.

## Plateformes

| Plateforme | Edge0 35B (moteur natif) | Edge0 145B, Edge0 120B (edge0-strata) |
| --- | --- | --- |
| macOS (Apple silicon) | Metal, CPU | — |
| iOS | Metal | — |
| Android | CPU | — |
| Linux | CPU | GPU NVIDIA (CUDA 13) |
| Windows | CPU, Vulkan | GPU NVIDIA (CUDA 13) |

Mesures : [Benchmarking](docs/benchmarking.md#reference-figures) pour Edge0 35B, et la page du modèle Edge0 145B et Edge0 120B (`docs/models/edge0-145b.md` dans le dépôt Edge0). Sur le moteur natif, le prefill et le decode peuvent s'exécuter sur des backends différents du même appareil, par exemple prefill sur Metal et decode sur CPU. Le 35B propose deux modes de routage : `predicted` (par défaut), où un pre-router permet au moteur de charger les experts du token suivant pendant le calcul du token courant, et `actual`, le calcul d'origine du modèle. Voir [Routing modes](docs/benchmarking.md#routing-modes).

## Démarrage rapide

Lancez les commandes depuis ce répertoire, `edge0-infer/` dans un clone du [dépôt Edge0](https://github.com/Edge0-AI/Edge0) :

```sh
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info
```

Téléchargez et convertissez ensuite les ressources du modèle, puis lancez une requête, une discussion ou un serveur local compatible OpenAI :

```sh
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Présente-toi en une phrase."
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

[Getting started](docs/getting-started.md) détaille chaque étape, y compris le téléchargement et la conversion des ressources, ainsi que les compilations pour Android, Windows et iOS.

Pour Edge0 145B et Edge0 120B, compilez edge0-strata puis lancez `edge0-infer serve --model <dossier du modèle>` ; les étapes sont dans [Getting started](docs/getting-started.md#edge0-145b-and-edge0-120b-edge0-strata).

## Documentation

- [Getting started](docs/getting-started.md) : compiler, obtenir les ressources du modèle, exécuter, discuter et servir, sur chaque plateforme
- [Edge0 145B](docs/edge0-145b.md) : exécuter ce modèle sur edge0-strata
- [Edge0 120B](docs/edge0-120b.md) : exécuter ce modèle sur edge0-strata
- [Configuration reference](docs/configuration.md) : paramètres du runtime, du chargement, du modèle et de la génération, leurs valeurs par défaut et les options de la CLI
- [Native API](docs/native-api.md) : utiliser le moteur dans le processus : cycle de vie, streaming, multi-tours, concurrence et annulation
- [OpenAI-compatible local server](docs/openai-compatible-server.md) : endpoints, champs de requête et limites
- [Architecture](docs/architecture.md) : les deux couches, l'organisation du moteur natif et le parcours d'une requête
- [Known limitations](docs/limitations.md) : comportements à connaître
- [Benchmarking](docs/benchmarking.md) : comment mesurer, réglages recommandés et chiffres de référence
- [Testing](docs/testing.md) : niveaux de test, ce dont chacun a besoin et comment les lancer
- [Model asset tools](tools/assets/README.md) : le répertoire des ressources, les vérifications, les petites fixtures de test et l'ajout d'un modèle
- [Integration examples](examples/README.md) : l'exemple de l'API native et le client du SDK OpenAI
- [Chat applications](examples/README.md#chat-applications) : exemples pour le bureau (macOS et Windows), Android et iOS
- [Measurement tools](tests/perf/README.md) : référence développeur des scripts de mesure et des rapports

## API native

L'API native correspond au moteur natif. Les applications se lient à la cible CMake `edge0::edge0` et incluent les en-têtes de [`include/edge0/`](include/edge0/), seule source des signatures ; les types et fonctions se trouvent dans l'espace de noms `edge0`. [`examples/native_chat.cpp`](examples/native_chat.cpp) montre le cycle de vie complet : ouvrir le runtime, charger un modèle, ouvrir une session, générer deux tours en streaming et tout libérer. Le [guide de l'API native](docs/native-api.md) explique le cycle de vie, le streaming, l'usage multi-tours, la concurrence, l'annulation et les erreurs.

## Licence et code tiers

Ce répertoire est distribué sous l'Apache License 2.0 du dépôt qui le contient. Le code tiers conserve sa propre licence :

- GGML, intégré sans modification dans `third_party/ggml/` (MIT, [`third_party/GGML_LICENSE`](third_party/GGML_LICENSE)) ;
- cpp-httplib (MIT, [`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)) et nlohmann/json (MIT, [`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)), utilisés uniquement par la CLI et le serveur ;
- le code du tokenizer adapté de llama.cpp dans `src/app/text/` (MIT, [`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)).

## Retours et contributions

Signalez les problèmes et proposez des modifications via les issues et les pull requests du dépôt qui contient ce répertoire. Le répertoire est développé dans un dépôt séparé et synchronisé ici : une modification acceptée est d'abord appliquée là-bas et arrive ici lors de la synchronisation suivante.
