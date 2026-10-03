# edge0 — Framework de Python

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

Este documento está dirigido a **desarrolladores que instalan y ejecutan el
framework de Python de edge0 desde el código fuente**. Cubre: inicio rápido
(instalación → modelos → ejecución), rendimiento y calidad medidos, y el
diseño detrás del stack.

edge0 se publica como **monorepo** —
[`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0) — cuyo nivel superior
contiene la documentación compartida (`docs/`) y los subproyectos de
plataforma (`python/` = este framework, además de `macos/`, `ios/`,
`android/`, `windows/`). El framework de Python es la implementación de
referencia de la receta edge0 — **SSD expert offload + Recover-LoRA +
predicción de enrutamiento del prerouter** — que ejecuta grandes modelos
sparse-MoE en Apple Silicon mediante MLX, con una memoria pico limitada por el
conjunto de expertos *activos* en lugar del número de parámetros.

Atribución de terceros: consulta [`NOTICE`](../NOTICE) en la raíz del
repositorio. Fichas de modelo:
[Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
· [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview).
Todos los comandos siguientes se ejecutan desde este directorio (`python/`).

---

## Inicio rápido

### Requisitos previos

| componente | requisito |
|---|---|
| SO / hardware | macOS en Apple Silicon (M1/M2/M3/M4) — el backend MLX es exclusivo de Apple Silicon; un backend CUDA está en la hoja de ruta |
| Python | 3.10+ (se recomienda 3.12) |
| MLX | `mlx==0.30.6` / `mlx-metal==0.30.6` con `mlx-lm==0.31.0` (consulta `pyproject.toml`) |
| Memoria | ~2.9 GB activos pico (edge0-35b), ~1.0 GB (edge0-8b), contextos cortos |
| Disco | ~23 GB (edge0-35b) / ~4.2 GB (edge0-8b); los pesos de los expertos se mapean con mmap y se leen bajo demanda, no se cargan en RAM por adelantado |

> Una salida ilegible o mezclada de idiomas en Apple A18 / A18 Pro indica un
> `mlx` antiguo: `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
> ([#8](https://github.com/Edge0-AI/Edge0/issues/8)).

### Instalación

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### Modelos

Ambos niveles están publicados en Hugging Face y ModelScope; cada repositorio
incluye el checkpoint base y los adaptadores entrenados de LoRA + prerouter en
**un solo directorio**, de modo que una única descarga es un modelo listo para
ejecutar:

```bash
# the repo's helper (defaults to the two published tiers):
.venv/bin/python scripts/fetch_models.py --tier edge0-8b  --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models

# or directly with the CLI:
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview  --local-dir models/edge0-8b
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview --local-dir models/edge0-35b
```

Apunta edge0 a un nivel mediante una variable de entorno, o pasa el directorio
directamente (el nivel se detecta automáticamente desde `config.json`):

```bash
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
```

### Ejecución

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

`python -m edge0 ...` es equivalente a `edge0 ...`.

### API de Python

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

`examples/demo.py` es el mismo recorrido mínimo (`edge0 demo` ejecuta
exactamente esta ruta).

---

## Rendimiento

Medido con `examples/bench.py` (prefill de un prompt de 3.3k tokens → 10 pasos
de calentamiento muestreados → 200 tokens de decode muestreados y
cronometrados, 2 ejecuciones por nivel):

| Nivel | Velocidad de decode | Throughput de prefill (frío / caliente)* | Memoria activa pico | Máquina de pruebas |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |

*Frío = primera petición tras iniciar el proceso (los pesos de expertos entran
por fallos de página desde el SSD); caliente = peticiones posteriores
(residentes en la caché de páginas).*

```bash
python examples/bench.py edge0-8b     # via $EDGE0_8B_MODEL
python examples/bench.py edge0-35b    # via $EDGE0_35B_MODEL
```

## Calidad

Ejecutado con [OpenCompass](https://github.com/open-compass/opencompass) bajo
configuraciones idénticas para los modelos edge0 (int4 + adaptadores
entrenados + prerouter) y las bases fp16 originales. El pipeline edge0 pierde
poco: **3.9 puntos de media para edge0-35b, 2.8 para edge0-8b** (MMLU-Pro
incluso supera a la base). Máximo 100:

| Benchmark | edge0-35b (int4) | Qwen3.6-35B-A3B (fp16) | edge0-8b (int4) | Ling 3.0 tiny (fp16) |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **Media** | **79.2** | **83.2** | **69.9** | **72.7** |

---

## Detalles técnicos

### Mecanismos principales

- **SSD expert offload** — los pesos de los expertos se transmiten desde el
  almacenamiento bajo demanda; la memoria pico está limitada por el conjunto
  activo, no por el número de parámetros.
- **Prerouter** — una cabeza entrenada predice el enrutamiento de expertos un
  paso por delante, de modo que las cargas de expertos se solapan con el
  forward pass en lugar de bloquearlo (**hasta +59%** de throughput de decode;
  la ganancia crece con la latencia del almacenamiento, el tamaño del modelo y
  el ancho de enrutamiento *K*).
- **Recover-LoRA** — la base int4 está congelada y los adaptadores LoRA se
  entrenan por destilación desde el teacher FP, recuperando la mayor parte de
  la pérdida de cuantización a 4 bits. Los adaptadores permanecen sin
  fusionar: una única base de solo lectura sirve a varios conjuntos de
  adaptadores.

### Diseño

- **Uso al estilo transformers** — `AutoModel` / `AutoConfig` / `AutoEngine`
  resuelven el nivel a partir del nombre del modelo.
- **Aislamiento de backends** — todo el código MLX vive en
  `src/edge0/backends/mlx/`; la lógica principal (especificaciones de modelos,
  prerouter, pool de expertos en streaming, servidor) depende solo de la
  fachada del backend (`backends/base.py`), de modo que un nuevo backend
  implementa la misma fachada (`backends/cuda/` es un espacio reservado) sin
  ningún cambio en el código principal.
- **Adaptadores como safetensors** — los pesos de LoRA y del prerouter son
  archivos `.safetensors` con metadatos de procedencia (fuente, versión, capas
  propietarias), resueltos desde el directorio del modelo o la alternativa
  `artifacts/` ignorada por git.
- **Modelo + adaptadores en un solo directorio** — un directorio de modelo
  contiene el checkpoint base y los adaptadores de ese modelo; actualizar los
  adaptadores solo reemplaza los archivos de adaptadores — la base permanece
  de solo lectura y nunca se fusiona.

### Estructura del paquete

```
src/edge0/
├── backends/mlx/                  # MLX backend (isolation boundary; cuda/ reserved)
├── engine/  models/  moe/  prerouter/  streaming/
├── adapters/  attention/  server/
└── cli.py  registry.py  sampling.py  config.py
```

---

## Documentación

- [Arquitectura](../docs/architecture.md)
- [Atención](../docs/attention.md) / [MoE](../docs/moe.md) / [Streaming SSD](../docs/streaming.md) / [prerouter](../docs/prerouter.md)
- [Añadir un modelo](../docs/adding-a-model.md)
- [edge0-35b](../docs/models/edge0-35b.md) / [edge0-8b](../docs/models/edge0-8b.md)
- Informe técnico: [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) ([PDF](../paper/main.pdf))

## Pruebas

```bash
pytest                 # unit tests (no real weights)
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q   # real-weight generation
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
```
