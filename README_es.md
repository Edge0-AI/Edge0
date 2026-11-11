<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**Un framework de inferencia MoE en streaming de código abierto — SSD expert offload + Recover-LoRA + predicción de enrutamiento del prerouter.**

**macOS** · **iOS** · **Android** · **Linux** · **Windows** — un programa para todos los modelos Edge0.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://edge0.ai/discord)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

</div>

## Novedades

- **[2026-11-13]** Lanzamos **`edge0-145b`** y **`edge0-120b`**, modelos latent mixture-of-experts derivados de Qwen3.8-Flash-Next para una sola GPU NVIDIA en Linux o Windows. Se ejecutan en [edge0-strata](edge0-strata/EDGE0.md), que `edge0-infer serve` y `edge0-infer chat` inician para ellos; consulta la [página del modelo](docs/models/edge0-145b.md).
- **[2026-11-04]** Lanzamos el **motor de inferencia unificado [`edge0-infer`](edge0-infer/README_es.md)**: construido sobre [GGML](https://github.com/ggml-org/ggml), un solo motor y una sola API nativa en C++ ejecutan el mismo modelo en macOS, iOS, Android y Windows, con una CLI, un servidor local compatible con OpenAI y ejemplos de aplicaciones de chat para escritorio, Android e iOS. La primera versión admite `edge0-35b`; `edge0-8b` llegará en una versión posterior. Sustituye al framework de Python y a los motores de cada plataforma; el código anterior se conserva en la etiqueta [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer).
- **[2026-09-30]** Lanzamos los **motores de inferencia de edge0 para cuatro plataformas — iOS, macOS, Android y Windows** (código fuente en la etiqueta [`pre-edge0-infer`](https://github.com/Edge0-AI/Edge0/tree/pre-edge0-infer)).
- **[2026-09-16]** Nuestro informe técnico está en arXiv: [The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction](https://arxiv.org/abs/2609.18063).
- **[2026-09-08]** Primera versión de código abierto de **edge0**, junto con los dos niveles de modelos — [`Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) y [`Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) — en Hugging Face y ModelScope.

## Acerca de

**edge0** es un framework de inferencia MoE en streaming de código abierto.
Generaliza la receta probada en producción — **SSD expert offload +
Recover-LoRA + predicción de enrutamiento del prerouter** — en un framework
extensible que ejecuta grandes modelos sparse-MoE en hardware de consumo: la
memoria pico está limitada por el conjunto de expertos *activos*, no por el
número de parámetros.

### Mecanismos principales

- **SSD expert offload**: los pesos de los expertos se transmiten desde el
  almacenamiento bajo demanda; la memoria pico está limitada por el conjunto
  activo, no por el número de parámetros.
- **Prerouter**: una cabeza entrenada predice el enrutamiento de expertos un
  paso por delante, de modo que las cargas de expertos se solapan con el
  forward pass en lugar de bloquearlo — **hasta +59%** de throughput de
  decode; la ganancia crece con la latencia del almacenamiento, el tamaño del
  modelo y el ancho de enrutamiento *K*.
- **Recover-LoRA**: la base int4 se mantiene congelada y los adaptadores LoRA
  se entrenan por destilación desde el teacher FP, recuperando la mayor parte
  de la pérdida de cuantización a 4 bits (consulta [Calidad](#calidad)). Los
  adaptadores se guardan aparte de la base y pueden activarse o desactivarse
  por separado.

### Plataformas

[`edge0-infer`](edge0-infer/README_es.md) es el programa que ejecutas (CLI, servidor local compatible con OpenAI y Anthropic, aplicaciones de chat), con los mismos comandos para todos los modelos. Cada modelo se ejecuta en el motor preparado para él, y edge0-infer elige el motor a partir de la carpeta del modelo; no hay nada que elegir: `edge0-35b` se ejecuta en el **motor nativo** de `edge0-infer/` (construido sobre GGML, con una API nativa en C++), y `edge0-145b` y `edge0-120b` en **edge0-strata**, en [`edge0-strata/`](edge0-strata/EDGE0.md), un fork de Strata.

| Plataforma | `edge0-35b` (motor nativo) | `edge0-145b`, `edge0-120b` (edge0-strata) |
|---|---|---|
| **macOS** (Apple Silicon) | Metal, CPU | — |
| **iOS** | Metal | — |
| **Android** | CPU | — |
| **Linux** | CPU | GPU NVIDIA (CUDA 13) |
| **Windows** | CPU, Vulkan | GPU NVIDIA (CUDA 13) |

Mediciones: [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures) para `edge0-35b` y la [página del modelo](docs/models/edge0-145b.md) para `edge0-145b` y `edge0-120b`. En el motor nativo, el prefill y el decode pueden ejecutarse en backends distintos del mismo dispositivo, por ejemplo prefill en Metal y decode en CPU. Los ejemplos de aplicaciones de chat para escritorio (macOS, Windows), Android e iOS están en [`edge0-infer/examples/`](edge0-infer/examples/README.md#chat-applications).

### Modelos

El framework incluye dos niveles de modelos, `edge0-35b` y `edge0-8b`. Cada nivel es una versión de
extremo a extremo: el checkpoint publicado, los adaptadores LoRA entrenados y
las cabezas prerouter entrenadas funcionan en conjunto como una sola unidad.

| Nivel | Checkpoint publicado | Perfil de inferencia | Motor |
|---|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4 bits, 40 capas, 256 expertos, prerouter K=4 | Motor nativo ✅ |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4 bits, 24 capas, 128 expertos, prerouter K=8 | Motor nativo (previsto) |
| [`edge0-145b`](docs/models/edge0-145b.md) | `<MODEL_CARD_URL>` | expertos latentes Q3_K o NVFP4 (1792), 48 capas, 512 expertos, K=10 | edge0-strata ✅ (Linux o Windows, GPU NVIDIA) |
| [`edge0-120b`](docs/models/edge0-145b.md) | `<MODEL_CARD_URL>` | expertos latentes Q3_K o NVFP4 (1280), 48 capas, 512 expertos, K=10 | edge0-strata ✅ (Linux o Windows, GPU NVIDIA) |

Los dos checkpoints publicados se construyen sobre modelos base sparse-MoE abiertos
(Qwen3.6-35B-A3B y el híbrido bailing Ling 3.0, respectivamente) e incluyen
el entrenamiento de LoRA y prerouter realizado para este framework.
edge0-infer descarga el checkpoint publicado con sus herramientas de recursos
y lo convierte en los recursos de modelo del motor (consulta
[Primeros pasos](#primeros-pasos)).
`edge0-145b` y `edge0-120b` forman un nivel latent-MoE aparte para una sola GPU NVIDIA; se ejecutan con edge0-strata, que `edge0-infer serve` y `edge0-infer chat` inician (consulta [edge0-145b y edge0-120b](docs/models/edge0-145b.md)).

### Diseño

- **Basado en GGML, el mismo código en todas las plataformas**: la biblioteca
  nativa solo depende de GGML, incluido sin modificaciones en una versión
  fija; la compilación no descarga nada;
- **Una API nativa**: las aplicaciones enlazan el target de CMake
  `edge0::edge0` e incluyen las cabeceras de `include/edge0/` para usar el
  motor dentro del proceso; la CLI de referencia y el servidor local
  usan los mismos ajustes y valores por defecto que la API nativa;
- **Expertos bajo demanda**: los expertos enrutados se leen del almacenamiento
  bajo demanda en un número fijo de ranuras por capa, de modo que un modelo
  más grande que el presupuesto de memoria del dispositivo se ejecuta sin
  cargarlo entero;
- **Dos modos de enrutamiento**: `predicted` (por defecto) usa el prerouter
  para preparar en segundo plano los expertos del siguiente token mientras se
  calcula el actual; `actual` es el cálculo original del modelo, para
  comparar;
- **Reutilización entre turnos**: en una conversación de varios turnos
  normalmente solo se calcula el mensaje nuevo de cada turno y se reutilizan
  los resultados anteriores.

Consulta la [arquitectura de edge0-infer](edge0-infer/docs/architecture.md).

### Calidad

Todos los benchmarks fueron ejecutados por nosotros con
[OpenCompass](https://github.com/open-compass/opencompass) bajo configuraciones
y parámetros idénticos tanto para los modelos edge0 (int4 + adaptadores
entrenados + enrutamiento prerouter) como para los modelos base fp16
originales. La pérdida del pipeline edge0 es pequeña: **3.9 puntos de media
para edge0-35b, 2.8 para edge0-8b** (MMLU-Pro incluso supera a la base).
Máximo 100:

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Media** | **79.2** | **83.2** | **69.9** | **72.7** |

`edge0-145b` y `edge0-120b` (**negrita** = mejor de la columna; consulta la [página del modelo](docs/models/edge0-145b.md#quality)):

| Model | deepswe 1.1 | terminal-bench 2.1 | AA-LCR | IFBench | GPQA-Diamond | HLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge0-145B** | **64.6** | **86.5** | 79.3 | 80.6 | 90.4 | 35.4 |
| **Edge0-120B** | 50.0 | 75.0 | **81.0** | 79.3 | 87.4 | 31.0 |
| DeepSeek-V4-Flash-0731 | 54.4 | 82.7 | 79.7 | 79.2 | 90.8 | 33.8 |
| Qwen3.8-Flash-Next | 58.7 | 83.3 | 79.1 | **81.3** | **91.7** | 35.9 |
| Qwen3.8-27B | 42.2 | 73.0 | **82.0** | 79.5 | 89.2 | 30.8 |
| Claude-Opus-4.6 (Max) | / | 78.2 | 78.0 | 62.5 | 91.3 | **40.0** |

### Benchmark

`edge0-35b` con enrutamiento `predicted`, medido con edge0-infer (cada rango va del mínimo al máximo de tres procesos):

| Plataforma | Dispositivo | Prompt | Primer token | Velocidad de decode | Pico de memoria |
|---|---|---|---|---|---|
| macOS Metal | Apple M5, 16 GiB | 3084 tokens | 5.60–5.87 s | 24.8–24.9 tok/s | 3.43 GiB (`phys_footprint`) |
| macOS CPU | Apple M5, 16 GiB | 627 tokens | 11.5–11.7 s | 21.7–22.3 tok/s | 2.96 GiB (`phys_footprint`) |
| iOS Metal | iPhone 17 Pro Max, 12 GB | 627 tokens | 8.20–8.26 s | 13.2 tok/s | 3.05 GiB (`phys_footprint`) |
| Android CPU | OnePlus PKR110, 16 GB | 627 tokens | 18.4–18.8 s | 11.4–11.8 tok/s | 2.82 GiB (RSS) |
| Windows CPU | Intel Core i5-12400, 32 GB | 627 tokens | 16.9–17.3 s | 7.3–8.2 tok/s | 2.86 GiB (conjunto de trabajo) |
| Windows Vulkan (ambas fases, 32 ranuras de expertos) | RTX 4070 SUPER, 12 GB | 627 tokens | 3.75–3.76 s | 23.6–24.0 tok/s | Aumento de memoria de GPU 5.35 GiB |

Las métricas de memoria difieren entre plataformas y no se pueden comparar directamente. El método de medición, las cifras con enrutamiento `actual` y los comandos para reproducirlas están en [Benchmarking](edge0-infer/docs/benchmarking.md#reference-figures).

`edge0-145b` y `edge0-120b` en edge0-strata, una GPU NVIDIA (tok/s):

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

Las filas `16 GiB VRAM + 20 GiB RAM` son la misma RTX 5090 D 24 GB, limitada por cgroup a 16 GiB de memoria de GPU y 20 GiB de memoria del sistema. Los detalles están en la [página del modelo](docs/models/edge0-145b.md#speed-measured-one-card).

## Primeros pasos

Ejecuta los siguientes comandos en `edge0-infer/`. Necesitas CMake 3.21 o
posterior, Ninja, un compilador de C++17/C11, uv y Git; el 35B necesita unos
18 GB para los pesos originales y unos 20 GB para los recursos convertidos,
más otro tanto de espacio libre durante la conversión.

### 1) Compilar

```bash
cd edge0-infer
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info        # debe mostrar una línea: model.0: edge0_35b
```

En máquinas sin Metal usa `--preset cpu`, `build/cpu/bin/edge0-infer` y
`--device cpu` en los comandos siguientes.

### 2) Obtener un modelo

**edge0-35b (motor nativo).** Descarga y convierte los recursos del modelo:

```bash
export MOE_SOURCE_35B=<directorio para los pesos originales>
export MOE_ASSETS_35B=<directorio para los recursos convertidos>

uv sync --project tools/assets
assets_py() { uv run --project tools/assets --offline --no-sync python -I "$@"; }

assets_py tools/assets/edge0_35b/fetch_source.py download --out "$MOE_SOURCE_35B"
assets_py tools/assets/edge0_35b/make_assets.py --source "$MOE_SOURCE_35B" --out "$MOE_ASSETS_35B"
assets_py tools/assets/check_assets.py check "$MOE_ASSETS_35B"
```

Se descarga `Edge0/Edge0-35B-A3B-preview` de Hugging Face y se verifica con
la revisión y los resúmenes fijados.

**edge0-145b y edge0-120b (edge0-strata).** Compila edge0-strata como indica [`edge0-strata/EDGE0.md`](edge0-strata/EDGE0.md) y usa una carpeta de modelo publicada (consulta [edge0-145b y edge0-120b](docs/models/edge0-145b.md)).

### 3) Ejecutar

```bash
# una instrucción
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Preséntate en una frase."
# chat de varios turnos
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
# servidor (compatible con OpenAI y Anthropic, por defecto http://127.0.0.1:8000)
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

Para edge0-145b y edge0-120b, `chat` y `serve` reciben la carpeta del modelo de la misma forma (`--model <carpeta del modelo>`); `run` es para el motor nativo.

Para las compilaciones de Android, Windows e iOS, y para usar el motor desde
tu propio proyecto CMake, consulta [Getting started](edge0-infer/docs/getting-started.md).

## Documentación

**Modelos**

- [Enrutamiento MoE y expertos](docs/moe.md) / [prerouter](docs/prerouter.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md) / [edge0-145b y edge0-120b](docs/models/edge0-145b.md)
- Informe técnico: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

**edge0-infer**

- [Getting started](edge0-infer/docs/getting-started.md): compilación, recursos del modelo, ejecución, chat y servidor en cada plataforma
- [Configuration](edge0-infer/docs/configuration.md) / [Native API](edge0-infer/docs/native-api.md) / [Servidor local](edge0-infer/docs/local-server.md)
- [Architecture](edge0-infer/docs/architecture.md) / [Known limitations](edge0-infer/docs/limitations.md)
- [Benchmarking](edge0-infer/docs/benchmarking.md) / [Testing](edge0-infer/docs/testing.md)
- [Herramientas de recursos del modelo y cómo añadir un modelo](edge0-infer/tools/assets/README.md)

**Motores**

- Motor nativo: [Architecture](edge0-infer/docs/architecture.md#native-engine) / [Native API](edge0-infer/docs/native-api.md)
- edge0-strata: [compilación y contrato de arranque](edge0-strata/EDGE0.md) / [edge0-145b](edge0-infer/docs/edge0-145b.md) y [edge0-120b](edge0-infer/docs/edge0-120b.md) en edge0-infer

## Hoja de ruta

### Q4 2026

**Plataformas y sistemas**

- ✅ **Motor de inferencia unificado de edge0 liberado** (2026-11-04) —
  [`edge0-infer`](edge0-infer/README_es.md): un solo motor y una sola API
  nativa para macOS, iOS, Android y Windows.
- **Compatibilidad con `edge0-8b` en edge0-infer**.

**Modelos y algoritmos**

Q4 trabaja en dos frentes: incorporar una arquitectura de próxima generación
al framework y convertir el razonamiento latente en un ahorro real de latencia
y no solo aritmético.

- **Soporte de arquitectura de próxima generación (clase Qwen3.8-Flash)** —
  ejecutar atención lineal híbrida (GDN + QSA), residual multirrama con
  puertas y topologías de embedding N-gram en edge0. Estos diseños encajan de
  forma natural con el streaming offload a SSD: la atención de estado O(1)
  evita que el thinking prolongado se convierta en un problema de KV cache, y
  las tablas N-gram de solo consulta se transmiten bajo demanda. Objetivo: que
  el nivel se ejecute en un solo dispositivo y que los benchmarks queden a una
  distancia aceptable de la base fp16.
- **Thinking latente + predicción anticipada de expertos por lotes** —
  convertir el razonamiento latente en un ahorro de *latencia*, no solo
  aritmético. El problema central de ingeniería: mover el enrutamiento de
  expertos de por posición a **una vez por bloque**, de modo que una sola
  predicción cubra cada posición y ronda de un bloque y **el volumen de carga
  de expertos se desacople del número de bucles de razonamiento** — además de
  prefetch entre bloques que carga los expertos del siguiente bloque dentro de
  la ventana de cómputo del bloque actual. El progreso se mide como **tiempo
  de la fase de thinking de extremo a extremo con la misma precisión** (nunca
  tokens/s).
- Más niveles de modelos y versiones de adaptadores sobre el pipeline
  existente.

## Contribuir

Las contribuciones son bienvenidas — issues, PRs, informes de benchmarks y
ports de modelos, todo cuenta.

La compilación y las pruebas se describen en [Testing](edge0-infer/docs/testing.md);
las comprobaciones que no necesitan recursos del modelo:

```bash
cd edge0-infer
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu -L "L0|L1"
```

Para cambios en `edge0-infer/`, abre issues y PRs en este repositorio como
de costumbre; el directorio se desarrolla en un repositorio aparte: los
mantenedores aplican primero allí los cambios aceptados, que llegan aquí con
la siguiente sincronización. Consulta [CONTRIBUTING](CONTRIBUTING.md).

Flujo de trabajo: fork → rama de feature → PR contra `main`. Mantén las
pruebas en verde y añade pruebas para los nuevos comportamientos.

## Cita

Si edge0 te resulta útil, cita nuestro informe técnico:

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

## Contacto

- **Discord**: [edge0.ai/discord](https://edge0.ai/discord) — chat y soporte de la comunidad
- **Email**: samuel@edge0.ai

Para bugs y solicitudes de funcionalidades, usa
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## Licencia

Apache-2.0; los componentes de terceros incluidos conservan sus propias licencias
(consulta [NOTICE](NOTICE)).
