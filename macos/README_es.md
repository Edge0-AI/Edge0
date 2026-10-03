# edge0

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

Inferencia en el dispositivo para Apple Silicon. La aplicación se comunica con un servidor local compatible con OpenAI que ejecuta modelos MoE de Edge0 (8B y 35B) con pesos de expertos en streaming, offload a SSD y Metal.

Requiere **macOS 14+** en **Apple Silicon (M3 o posterior)**. Todos los comandos siguientes se ejecutan desde la raíz de este directorio. MLX v0.30.6 ya está en `third_party/mlx`.

## Inicio rápido

### Requisitos previos

- Xcode Command Line Tools (compilador de Metal)
- CMake 3.24+
- Rust 1.88 (`rustup` elegirá la versión desde `rust-toolchain.toml`)
- Node 22 o posterior (se usa yarn 1.22 si ya está instalado; en caso contrario `check-prereqs` lo habilita mediante Corepack)

`make app` ejecuta esta verificación. Para comprobarlo antes:

```bash
bash scripts/check-prereqs.sh
```

### Compilación y empaquetado

```bash
make app
open dist/edge0-0.1.0-arm64.app
```

`make app` instala las dependencias de JavaScript, compila `edge0`, `edge0d` y `edge0-engine`, compila el motor nativo y después genera `dist/edge0-<version>-arm64.app`. No inicia la aplicación ni el servicio local. El `.app` está firmado de forma ad-hoc.

### Primer inicio

1. Abre el `.app`. El servicio local (`edge0d` en `127.0.0.1:8000`) se inicia con él.
2. En la página Modelos, descarga un modelo desde Hugging Face (se guarda en `~/.edge0`), cárgalo y chatea.
3. Chat: Enter envía. Las respuestas se transmiten en Markdown, incluidas las fórmulas matemáticas `$inline$` y `$$block$$`.
4. API HTTP (loopback no necesita token):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

El acceso fuera de loopback requiere `Authorization: Bearer <token>` desde la página Servicio.

### Desarrollo

Para iterar sin empaquetar:

```bash
make setup
cargo build -p edge0d -p edge0-engine
make engine
cd app && yarn tauri dev
```

Esto inicia una ventana de desarrollo que puede lanzar el servicio local. `make engine` compila la biblioteca nativa de MLX/Metal.

### Uso de la aplicación

| Pantalla | Qué hace |
| --- | --- |
| Chat | Chat en streaming, prompts predefinidos, estilo de generación (Preciso / Equilibrado / Creativo), Markdown + fórmulas matemáticas |
| Modelos | Descargar, cargar/liberar, keep-alive |
| Servicio | Iniciar/detener el demonio, bind en LAN, LaunchAgent, token de API, verificación del entorno, registro de peticiones |
| Configuración | Tema, tamaño de fuente, system prompt opcional |
| Barra de menús | Estado, Abrir ventana, Reiniciar servicio, Salir (detiene el demonio que inició esta aplicación) |

Los datos viven en `~/.edge0` (modelos, base de datos de sesiones, logs). La CLI es `Contents/Resources/bin/edge0` dentro del `.app`. `edge0 uninstall` detiene el servicio y cualquier LaunchAgent; añade `--purge` para eliminar los datos de usuario tras confirmación.

## Rendimiento

Medido en **MacBook Air, Apple M3**.

| Modelo | Decode | Prefill | Memoria residente |
| --- | --- | --- | --- |
| Edge0-35B-A3B | 10–12 tok/s | ~70–130 tok/s | 4 GB |
| Edge0-8B-A1B | 18–20 tok/s | ~300–570 tok/s | 1.7 GB |

## Detalles técnicos

```
edge0.app (Tauri 2 + React 19)
    HTTP 127.0.0.1:8000  →  edge0d (OpenAI-compatible + /v1/edge0/*)
    Unix socket frames   →  edge0-engine
    dlopen               →  libedge0_engine_native.dylib + libmlx.dylib
```

- **UI:** Tauri 2, React 19, TanStack Router, Tailwind, Vercel AI SDK, streamdown + `@streamdown/math` (KaTeX).
- **Demonio:** Rust. `/v1/chat/completions` (SSE). Descarga desde Hugging Face por defecto.
- **Motor:** C++/Metal sobre mlx 0.30.6. Expertos en streaming int4 (slices mmap + pila caliente LRU + SSD offload), prefetch del prerouter, LoRA sin fusionar, caché de prefijos KV. NAX GEMM está desactivado por defecto (`MLX_METAL_NO_NAX`) porque mlx 0.30.6 es numéricamente incorrecto para edge0-8b en M5.

`EDGE0_HOME` tiene por defecto `~/.edge0`.
