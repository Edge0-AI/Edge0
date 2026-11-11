<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Leistungsstarke KI auf deinem eigenen Computer.</h3>
<p align="center">Chatten, programmieren und mit Bildern arbeiten, lokal.</p>

<p align="center">
  <a href="#modelle">Modelle</a> · <a href="#installation"><b>Installation</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#hilfe">Hilfe</a> · <a href="#strata-unterstützen">Unterstützen</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Strata herunterladen"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT-Lizenz"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Strata auf Buy Me a Coffee unterstützen"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <b>Deutsch</b> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Kostenlos, Open Source und ganz in deiner Hand.** Nutze Strata auf dem Desktop oder mit deinen liebsten Coding-Tools. Kein Konto nötig.

## Modelle

Wir empfehlen **zuerst ISTA-DASLab**, danach **Unsloth** und andere Anbieter, soweit ein passendes Modell in deinen Speicher passt.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Anbieter | Auswahl |
| --- | --- |
| **ISTA-DASLab · empfohlen** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) zum Programmieren. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Geringerer Speicherbedarf, aber langsamer. |
| **Weitere Optionen** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Größen, Speicherbedarf und Kompatibilität vergleichen](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Das sind die derzeit dokumentierten Pakete für dieses Modell.

[Modellanleitung und Kompatibilität](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In der **Desktop-App** und im **Terminal**. Was nicht in den Arbeitsspeicher passt, wird von deiner SSD gelesen. Darum brauchst du viel freien Speicherplatz.

| Set (IST-DAS) | Download | Freier Speicherplatz nötig |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | etwa 245 GB |
| 2.00 bpw | 257 GB | etwa 265 GB |
| 2.25 bpw | 274 GB | etwa 280 GB |

Läuft auf NVIDIA RTX 30, 40 und 50 (Windows und Linux) und auf AMD Radeon AI PRO R9700 (Linux). Das Setup lädt die Dateien und die DeepSeek-Engine herunter und startet dann das Modell.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Gemessen mit dem 2.00-bpw-Set auf Strata 0.2.5. „Ausgabe“ ist das Tempo der Antwort, „Prompt“ zeigt, wie schnell Strata deine Eingabe liest.

| Hardware | Ausgabe | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 bis 6 Token/s | etwa 110 Token/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 bis 11 Token/s | etwa 155 Token/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 bis 25 Token/s | 320 bis 360 Token/s (8K) |
| 4x Radeon AI PRO R9700 | 74 bis 77 Token/s | 400 bis 450 Token/s (8K) |

[Speicher, Festplatte und Details](docs/DEEPSEEK41.md) · [Versionshinweise](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Unsicher, was passt? [Wähle ein Modell für deinen Computer](docs/MODELS.md#pick-by-memory).

## Installation

<a id="option-a-the-strata-app"></a>

### Desktop-App

**[Strata für Windows, Linux oder Mac herunterladen →](https://github.com/Niko1221/Strata/releases/latest)**

Installiere die App, wähle ein Modell und leg mit dem Chat los. Es gibt einen Installer pro System. Beim ersten Start lädt die App die Engine für deine Grafikkarte herunter. [Hilfe zum Installer](docs/INSTALL.md#the-app).

<sub>Dateien: `Strata-0.2.9-windows-x64-setup.exe` · `Strata-0.2.9-linux-x86_64.AppImage` · `Strata-0.2.9-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata-Desktop-App mit Chat und Live-Codevorschau nebeneinander">
  <br><sub>Chat, Dateien und Live-Vorschau in einem Fenster.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal-App

Lieber im Terminal? Installiere Strata und starte dann **`strata launch`**.

<details open>
<summary><b>Installationsbefehle</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Öffne unter Windows danach ein neues Terminal. Dann:

```sh
strata launch
```

Folge den Anweisungen, um ein Modell herunterzuladen und zu starten. [Terminal-Anleitung](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata-Terminal-App mit Modellaktivität, Tempo, Speicher und letzten Anfragen">
  <br><sub>Behalte Modell und Hardware im Blick, während es läuft.</sub>
</p>

<a id="build-from-source"></a>

[Aus dem Quellcode bauen](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Vollständige Installationsanleitung](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Unterstützte Hardware | Anleitung |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [NVIDIA-Setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD-Setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimentell** | [RX-6000-Setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel-Setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 oder neuer** | [Mac-Setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix-Halo-Setup](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, und weitere über die OpenAI- und Anthropic-kompatiblen APIs.

Führe **`strata connect`** aus, um die Einrichtung für dein Tool zu sehen. [Verbindungsanleitung](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Hilfe

[Alle Anleitungen](docs) · [Fehlerbehebung](docs/TROUBLESHOOTING.md) · [Fehler melden](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [So funktioniert es](docs/HOW_IT_WORKS.md) · [Mitmachen](CONTRIBUTING.md)

## Strata unterstützen

**Hilf mit, lokale KI für alle offen zu halten.** Ein Kaffee hilft, Entwicklung und Tests auf mehr Hardware zu finanzieren.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: Strata unterstützen" height="48"></a>

[MIT-Lizenz](LICENSE) · [Danksagung und Modelllizenzen](docs/HOW_IT_WORKS.md#credits) · [Datenschutz](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Für KI-Tools und automatische Leser: <a href="AGENTS.md">technische Referenz</a> · <a href="docs/AI_SETUP.md">Setup-Anleitung</a>.</sub>

**Sponsoring, Hardware-Partnerschaften oder Zusammenarbeit:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
