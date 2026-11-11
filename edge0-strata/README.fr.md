<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Une IA puissante sur votre propre ordinateur.</h3>
<p align="center">Discutez, programmez et travaillez avec des images, en local.</p>

<p align="center">
  <a href="#modèles">Modèles</a> · <a href="#installation"><b>Installation</b></a> ·
  <a href="#matériel">Matériel</a> · <a href="#outils">Outils</a> ·
  <a href="#aide">Aide</a> · <a href="#soutenir-strata">Soutenir</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Télécharger Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licence MIT"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Soutenir Strata sur Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <b>Français</b> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Gratuit, open source, et à vous de l'exécuter.** Utilisez Strata sur votre bureau ou avec vos outils de programmation préférés. Aucun compte nécessaire.

## Modèles

Nous recommandons **d'abord ISTA-DASLab**, puis **Unsloth** et d'autres fournisseurs, lorsqu'un modèle compatible tient en mémoire.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Fournisseur | Choix |
| --- | --- |
| **ISTA-DASLab · recommandé** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder : IQ1\_M](docs/MODELS.md#coder) pour programmer. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Utilise moins de mémoire, mais plus lent. |
| **Autres options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Comparer les tailles, la mémoire nécessaire et la compatibilité](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Ce sont les paquets actuellement documentés pour ce modèle.

[Guide du modèle et compatibilité](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

Dans l'**application de bureau** et dans le **Terminal**. Ce qui ne tient pas en mémoire est lu depuis votre SSD : il faut donc beaucoup d'espace disque libre.

| Jeu (IST-DAS) | Téléchargement | Espace disque libre nécessaire |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | environ 245 GB |
| 2.00 bpw | 257 GB | environ 265 GB |
| 2.25 bpw | 274 GB | environ 280 GB |

Fonctionne sur les NVIDIA RTX 30, 40 et 50 (Windows et Linux) et sur l'AMD Radeon AI PRO R9700 (Linux). L'installation télécharge les fichiers et le moteur DeepSeek, puis démarre le modèle.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Mesuré avec le jeu 2.00 bpw sur Strata 0.2.5. « Sortie » est la vitesse de la réponse, « Prompt » la vitesse à laquelle Strata lit votre texte.

| Matériel | Sortie | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB de RAM | 4 à 6 tokens/s | environ 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB de RAM | 9 à 11 tokens/s | environ 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB de RAM | 21 à 25 tokens/s | 320 à 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 à 77 tokens/s | 400 à 450 tokens/s (8K) |

[Mémoire, disque et détails](docs/DEEPSEEK41.md) · [Notes de version](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Vous ne savez pas ce qui tient ? [Choisir un modèle pour votre ordinateur](docs/MODELS.md#pick-by-memory).

## Installation

<a id="option-a-the-strata-app"></a>

### Application de bureau

**[Télécharger Strata pour Windows, Linux ou Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Installez l'application, choisissez un modèle et commencez à discuter. Il y a un seul installateur par système. Au premier lancement, l'application télécharge le moteur de votre carte graphique. [Aide pour l'installateur](docs/INSTALL.md#the-app).

<sub>Fichiers: `Strata-0.2.9-windows-x64-setup.exe` · `Strata-0.2.9-linux-x86_64.AppImage` · `Strata-0.2.9-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Application de bureau Strata avec une discussion et un aperçu du code en direct côte à côte">
  <br><sub>Discussion, fichiers et aperçu en direct dans une seule fenêtre.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Application Terminal

Vous préférez le terminal ? Installez Strata, puis lancez **`strata launch`**.

<details open>
<summary><b>Commandes d'installation</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Sous Windows, ouvrez ensuite un nouveau terminal. Puis :

```sh
strata launch
```

Suivez les invites pour télécharger et démarrer un modèle. [Guide du Terminal](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Application Terminal de Strata affichant l'activité du modèle, la vitesse, la mémoire et les requêtes récentes">
  <br><sub>Suivez votre modèle et votre matériel pendant qu'il tourne.</sub>
</p>

<a id="build-from-source"></a>

[Compiler depuis les sources](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Guide d'installation complet](docs/INSTALL.md)

<a id="what-you-need"></a>

## Matériel

| Matériel pris en charge | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Installation NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Installation AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · expérimental** | [Installation RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Installation Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 ou plus récent** | [Installation sur Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Installation Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Outils

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, et d'autres via les API compatibles OpenAI et Anthropic.

Lancez **`strata connect`** pour voir la configuration de votre outil. [Guide de connexion](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Aide

[Tous les guides](docs) · [Dépannage](docs/TROUBLESHOOTING.md) · [Signaler un bogue](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [Comment ça marche](docs/HOW_IT_WORKS.md) · [Contribuer](CONTRIBUTING.md)

## Soutenir Strata

**Aidez à garder l'IA locale ouverte à tous.** Un café aide à financer le développement et les tests sur davantage de matériel.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee : soutenir Strata" height="48"></a>

[Licence MIT](LICENSE) · [Crédits et licences des modèles](docs/HOW_IT_WORKS.md#credits) · [Confidentialité](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Pour les outils d'IA et les lecteurs automatiques : <a href="AGENTS.md">référence technique</a> · <a href="docs/AI_SETUP.md">instructions d'installation</a>.</sub>

**Parrainages, partenariats matériel ou collaborations :** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
