<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">IA poderosa no seu próprio computador.</h3>
<p align="center">Converse, programe e trabalhe com imagens, localmente.</p>

<p align="center">
  <a href="#modelos">Modelos</a> · <a href="#instalação"><b>Instalação</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#ferramentas">Ferramentas</a> ·
  <a href="#ajuda">Ajuda</a> · <a href="#apoie-o-strata">Apoiar</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Baixar o Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="Licença MIT"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Apoie o Strata no Buy Me a Coffee"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b> · <a href="README.pl.md">Polski</a></sub></p>

**Gratuito, de código aberto e seu para executar.** Use o Strata no desktop ou com suas ferramentas de programação favoritas. Não precisa de conta.

## Modelos

Recomendamos **primeiro a ISTA-DASLab**, depois a **Unsloth** e outros provedores, quando um modelo compatível cabe na memória.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provedor | Opções |
| --- | --- |
| **ISTA-DASLab · recomendado** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) para programar. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Usa menos memória, mas é mais lento. |
| **Outras opções** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare tamanhos, necessidades de memória e compatibilidade](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. Esses são os pacotes documentados atualmente para este modelo.

[Guia do modelo e compatibilidade](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

No **aplicativo de desktop** e no **Terminal**. O que não cabe na memória é lido do seu SSD, por isso é preciso muito espaço livre em disco.

| Conjunto (IST-DAS) | Download | Espaço livre necessário |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | cerca de 245 GB |
| 2.00 bpw | 257 GB | cerca de 265 GB |
| 2.25 bpw | 274 GB | cerca de 280 GB |

Roda em NVIDIA RTX 30, 40 e 50 (Windows e Linux) e em AMD Radeon AI PRO R9700 (Linux). A instalação baixa os arquivos e o motor do DeepSeek e depois inicia o modelo.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Medido com o conjunto de 2.00 bpw no Strata 0.2.5. “Saída” é a velocidade da resposta; “Prompt” é a rapidez com que o Strata lê o seu texto.

| Hardware | Saída | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB de RAM | 4 a 6 tokens/s | cerca de 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB de RAM | 9 a 11 tokens/s | cerca de 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB de RAM | 21 a 25 tokens/s | 320 a 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 a 77 tokens/s | 400 a 450 tokens/s (8K) |

[Memória, disco e detalhes](docs/DEEPSEEK41.md) · [Notas da versão](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Não sabe o que cabe? [Escolha um modelo para o seu computador](docs/MODELS.md#pick-by-memory).

## Instalação

<a id="option-a-the-strata-app"></a>

### Aplicativo de desktop

**[Baixar o Strata para Windows, Linux ou Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Instale o aplicativo, escolha um modelo e comece a conversar. Há um instalador por sistema. Na primeira vez que abre, o aplicativo baixa o motor da sua placa de vídeo. [Ajuda com o instalador](docs/INSTALL.md#the-app).

<sub>Arquivos: `Strata-0.2.9-windows-x64-setup.exe` · `Strata-0.2.9-linux-x86_64.AppImage` · `Strata-0.2.9-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Aplicativo de desktop do Strata com um chat e uma prévia do código ao vivo lado a lado">
  <br><sub>Chat, arquivos e prévia ao vivo em uma só janela.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Aplicativo de Terminal

Prefere o terminal? Instale o Strata e execute **`strata launch`**.

<details open>
<summary><b>Comandos de instalação</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

No Windows, abra um novo terminal depois. Em seguida:

```sh
strata launch
```

Siga as instruções para baixar e iniciar um modelo. [Guia do Terminal](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Aplicativo de Terminal do Strata mostrando a atividade do modelo, a velocidade, a memória e as solicitações recentes">
  <br><sub>Acompanhe seu modelo e seu hardware enquanto ele roda.</sub>
</p>

<a id="build-from-source"></a>

[Compilar a partir do código-fonte](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Guia de instalação completo](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Hardware compatível | Guia |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50** | [Instalação para NVIDIA](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [Instalação para AMD](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [Instalação para RX 6000](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Instalação para Intel](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 ou mais novo** | [Instalação para Mac](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Instalação para Strix Halo](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Ferramentas

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, e outras, pelas APIs compatíveis com OpenAI e Anthropic.

Execute **`strata connect`** para ver as instruções da sua ferramenta. [Guia de conexão](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Ajuda

[Todos os guias](docs) · [Solução de problemas](docs/TROUBLESHOOTING.md) · [Relatar um bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [Como funciona](docs/HOW_IT_WORKS.md) · [Contribuir](CONTRIBUTING.md)

## Apoie o Strata

**Ajude a manter a IA local aberta para todos.** Um café ajuda a financiar o desenvolvimento e os testes em mais hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: apoie o Strata" height="48"></a>

[Licença MIT](LICENSE) · [Créditos e licenças dos modelos](docs/HOW_IT_WORKS.md#credits) · [Privacidade](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>Para ferramentas de IA e leitores automáticos: <a href="AGENTS.md">referência técnica</a> · <a href="docs/AI_SETUP.md">instruções de instalação</a>.</sub>

**Patrocínios, parcerias de hardware ou colaborações:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).
