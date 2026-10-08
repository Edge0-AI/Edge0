<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**Un framework de inferencia MoE en streaming de código abierto — SSD expert offload + Recover-LoRA + predicción de enrutamiento del prerouter.**

**Python** · **macOS** · **iOS** · **Android** — una receta, todos los dispositivos.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--35B--A3B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
[![ModelScope](https://img.shields.io/badge/ModelScope-Edge0--8B--A1B--preview-624AFF?style=for-the-badge)](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)
[![arXiv](https://img.shields.io/badge/arXiv-2609.18063-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.18063)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

</div>

## Novedades

- **[2026-09-30]** Lanzamos los **motores de inferencia de edge0 para cuatro plataformas — iOS, macOS, Android y Windows** — de modo que los usuarios obtienen la mejor experiencia de inferencia en todas las arquitecturas y plataformas. El código fuente se ha liberado en este repositorio ([`ios/`](ios/) · [`macos/`](macos/) · [`android/`](android/) · [`windows/`](windows/)) — consulta el README de cada directorio para más detalles. El **framework de inferencia unificado** llegará en **Q4 2026**; consulta la [Hoja de ruta](#hoja-de-ruta).
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
  adaptadores permanecen sin fusionar: una única base de solo lectura sirve a
  varios conjuntos de adaptadores.

### Plataformas

Un repositorio, una receta, runtimes por plataforma:

| Plataforma | Directorio | Stack | Estado |
|---|---|---|---|
| **Python** (macOS · Apple Silicon) | [`python/`](python/README_es.md) | Python + MLX | ✅ Disponible ahora |
| Aplicación y CLI para **macOS** | [`macos/`](macos/README_es.md) | Rust | ✅ Código abierto (2026-09-30) |
| Aplicación para **iOS** | [`ios/`](ios/README_es.md) | Swift + MLX Swift | ✅ Código abierto (2026-09-30) |
| Aplicación y motor para **Android** | [`android/`](android/README_es.md) | Kotlin + motor nativo | ✅ Código abierto (2026-09-30) |
| Aplicación y motor para **Windows** | [`windows/`](windows/README_es.md) | C++ + HIP (ROCm) / Vulkan | ✅ Código abierto (2026-09-30) |

### Modelos

El framework incluye dos niveles de modelos. Cada nivel es una versión de
extremo a extremo: el checkpoint publicado, los adaptadores LoRA entrenados y
las cabezas prerouter entrenadas funcionan en conjunto como una sola unidad.

| Nivel | Checkpoint publicado | Perfil de inferencia |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview) | 4 bits, 40 capas, 256 expertos, prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) · [ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview) | 4 bits, 24 capas, 128 expertos, prerouter K=8 |

Ambos checkpoints se construyen sobre modelos base sparse-MoE abiertos
(Qwen3.6-35B-A3B y el híbrido bailing Ling 3.0, respectivamente) e incluyen
el entrenamiento de LoRA y prerouter realizado para este framework — los
archivos de adaptadores están junto a cada checkpoint y se cargan
automáticamente, de modo que `edge0 serve <nivel>` ejecuta el pipeline
entrenado sin ninguna configuración adicional.

### Diseño

- **Uso al estilo transformers**: `AutoModel` / `AutoConfig` / `AutoEngine`
  resuelven el nivel a partir del nombre del modelo;
- **Aislamiento de backends (framework de Python)**: dentro del framework de
  Python, todo el código MLX vive en `python/src/edge0/backends/mlx/`; la
  lógica principal (especificaciones de modelos, prerouter, pool de expertos
  en streaming, servidor) depende solo de la fachada del backend
  (`backends/base.py`), de modo que un nuevo backend implementa la misma
  fachada (`backends/cuda/` es un espacio reservado) sin ningún cambio en el
  código principal. Los motores de iOS / macOS / Android incluyen hoy stacks
  nativos de cada plataforma — llevar todas las plataformas bajo una única
  capa de acceso es exactamente lo que entregará el framework de inferencia
  unificado (consulta la [Hoja de ruta](#hoja-de-ruta));
- **Adaptadores como safetensors**: los pesos de LoRA y del prerouter son
  archivos `.safetensors` con metadatos de procedencia (fuente, versión, capas
  propietarias), resueltos desde el directorio del modelo o `artifacts/`;
- **Modelo + adaptadores en un solo directorio**: un directorio de modelo
  contiene tanto el checkpoint base (`config.json` / `model*.safetensors` /
  tokenizer) como los adaptadores de ese modelo; actualizar los adaptadores
  solo reemplaza los archivos de adaptadores — la base permanece de solo
  lectura y nunca se fusiona.

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

### Benchmark

Medido con `python/examples/bench.py` (prefill de un prompt de 3.3k tokens →
10 pasos de calentamiento muestreados → 200 tokens de decode muestreados y
cronometrados, 2 ejecuciones por nivel):

| Nivel | Velocidad de decode | Throughput de prefill (frío / caliente)* | Memoria activa pico | Máquina de pruebas |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*Frío = primera petición tras iniciar el proceso (los pesos de expertos entran
por fallos de página desde el SSD); caliente = peticiones posteriores
(residentes en la caché de páginas). Las cifras de prefill son throughput sobre
un prompt de ~3.3k tokens (`BENCH_LONG=1`).*

Reproducir:

```bash
cd python
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
```

## Primeros pasos

### Python (macOS · Apple Silicon)

#### Requisitos

- **SO / hardware**: el backend MLX se ejecuta en macOS con Apple Silicon
  (M1/M2/M3/M4). El backend CUDA está en la hoja de ruta — el framework de
  Python aún no admite otras plataformas.
- **Python**: 3.10+ (se recomienda 3.12).
- **MLX**: `mlx==0.30.6` / `mlx-metal==0.30.6` con `mlx-lm==0.31.0` (consulta
  `python/pyproject.toml`). Una salida ilegible o mezclada de idiomas en Apple
  A18 / A18 Pro indica un `mlx` antiguo: `pip install 'mlx==0.30.6'
  'mlx-metal==0.30.6'` ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).
- **Memoria**: ~2.9 GB de memoria activa pico para `edge0-35b`, ~1.0 GB para
  `edge0-8b` (contextos cortos; consulta [Benchmark](#benchmark)). Añade
  margen para el SO, el tokenizer y el crecimiento de la KV cache en contextos
  largos.
- **Disco**: los checkpoints de 4 bits ocupan ~23 GB (`edge0-35b`) y ~4.2 GB
  (`edge0-8b`); los pesos de los expertos se mapean con mmap y se leen bajo
  demanda, no se cargan en RAM por adelantado.

#### 1) Instalación

```bash
cd python
# Python >= 3.10; the MLX backend requires macOS with Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

#### 2) Descargar un modelo

Los dos niveles están publicados en Hugging Face y ModelScope — cada
repositorio incluye el checkpoint base y los adaptadores entrenados de LoRA +
prerouter en **un solo directorio**, de modo que una única descarga es un
modelo listo para ejecutar:

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) (~23 GB) · [mirror en ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-35B-A3B-preview)
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) (~4.2 GB) · [mirror en ModelScope](https://www.modelscope.cn/models/Edge0/Edge0-8B-A1B-preview)

```bash
# with the repo's helper (defaults to the two repos above):
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview     --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview     --local-dir models/edge0-8b
```

De cualquier forma obtendrás un directorio así:

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer files   # base checkpoint
├── lora_edge0_35b.safetensors          # trained LoRA adapters
└── prerouter_edge0_35b.safetensors     # trained prerouter heads
```

#### 3) Apuntar edge0 al modelo

Los nombres de nivel se resuelven a directorios locales mediante variables de
entorno (dónde guardes la descarga depende de ti):

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

O sáltate las variables de entorno y pasa el directorio directamente — el
nivel se detecta automáticamente desde el `config.json` del checkpoint:

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
```

#### 4) Ejecutar

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

`python -m edge0 ...` es equivalente a `edge0 ...`.

#### API de Python

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

`python/examples/demo.py` es el mismo recorrido mínimo (`edge0 demo` ejecuta
exactamente esta ruta).

#### Modelos y adaptadores

- **Checkpoint**: el directorio original del modelo (`config.json`,
  `model*.safetensors`, tokenizer). `edge0 serve <dir>` /
  `AutoEngine.from_pretrained(<dir>)` detectan el nivel desde `config.json`.
- Los **adaptadores** (LoRA + prerouter, safetensors) se resuelven
  automáticamente desde cualquiera de estas ubicaciones:
  - el directorio del modelo (recomendado): junto a la base, p. ej.
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`;
  - `artifacts/` en la raíz del proyecto Python (ignorado por git): una caché
    alternativa opcional para los safetensors de adaptadores que no estén
    junto al modelo.
- Los repositorios de modelos publicados incluyen tanto el checkpoint base
  como la versión actual por defecto de los adaptadores, de modo que
  `scripts/fetch_models.py` produce un directorio de modelo listo para
  ejecutar. Consulta la página de documentación de cada modelo para conocer la
  procedencia de sus adaptadores (datos de entrenamiento, disposición de las
  capas propietarias).
- Ambos adaptadores son necesarios para el pipeline prerouter + LoRA; si falta
  un archivo, `edge0` falla con un mensaje claro (o pasa `--no-prerouter` /
  `--no-lora` para ejecutar el modelo base simple).

#### Documentación

- [Arquitectura](docs/architecture.md)
- [Atención](docs/attention.md) / [MoE](docs/moe.md) / [Streaming SSD](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [Añadir un modelo](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md)
- Informe técnico: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](paper/main.pdf))

### macOS / iOS / Android / Windows

Los cuatro motores de plataforma son de código abierto en este repositorio —
más detalles en el README de cada directorio:

- **macOS**: CLI / demonio / aplicación de escritorio local (Rust) — consulta [`macos/README_es.md`](macos/README_es.md)
- **iOS**: aplicación para iPhone en el dispositivo (Swift + MLX Swift) — consulta [`ios/README_es.md`](ios/README_es.md)
- **Android**: aplicación en el dispositivo + motor nativo (Kotlin) — consulta [`android/README_es.md`](android/README_es.md)
- **Windows**: aplicación de escritorio + motor nativo (C++ + HIP (ROCm) / Vulkan) — consulta [`windows/README_es.md`](windows/README_es.md)

El **framework de inferencia unificado** — una única capa de acceso, con el
runtime adaptándose automáticamente a iOS / macOS / Android / Windows /
Python — llegará en **Q4 2026**; consulta la [Hoja de ruta](#hoja-de-ruta).

## Hoja de ruta

### Q4 2026

**Plataformas y sistemas**

- **Framework de inferencia unificado de edge0** — liberaremos
  un framework de inferencia unificado: **una única capa de acceso
  unificada** (una sola API para chat / serve / uso en el dispositivo), con el
  **runtime adaptándose automáticamente a la plataforma de hardware** — iOS,
  macOS, Android, Windows y Python. Se construye sobre los motores de
  plataforma ya liberados en este repositorio (`ios/` · `macos/` ·
  `android/` · `windows/`).
- **Backend CUDA** para el framework de Python — espacio reservado en
  `python/src/edge0/backends/cuda/`, el código principal no necesita ningún
  cambio.

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

**Framework de Python** (disponible ahora):

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

CI ejecuta pruebas unitarias (macOS + MLX) y una suite de higiene del
repositorio (sin rutas hardcodeadas, checks de límites de backend y de
secretos) en cada PR.

**Runtimes de plataforma** (macOS / iOS / Android / Windows): cada
directorio de plataforma incluye su propia guía de compilación y pruebas
— consulta el README de cada directorio.

Flujo de trabajo: fork → rama de feature → PR contra `main`. Mantén la suite
de higiene en verde y añade pruebas para los nuevos comportamientos.

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

Los canales de comunidad y soporte llegarán pronto — esta sección enumerará
las vías oficiales para contactarnos:

- **Email**: samuel@edge0.ai

Para bugs y solicitudes de funcionalidades, usa
[GitHub Issues](https://github.com/Edge0-AI/Edge0/issues).

## Licencia

Apache-2.0, incluido el código de terceros incorporado (consulta
[NOTICE](NOTICE)).
