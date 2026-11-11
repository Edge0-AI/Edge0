# edge0-infer

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

`edge0-infer` ejecuta modelos de lenguaje de mezcla de expertos (MoE) en tu propio hardware: una CLI, un servidor local compatible con OpenAI y aplicaciones de chat, con los mismos comandos para todos los modelos. Cada modelo se ejecuta en el motor preparado para él, y `edge0-infer` elige el motor a partir de la carpeta del modelo; no hay nada que elegir:

- **Edge0 35B** (Qwen3.6-35B-A3B con un LoRA y un pre-router entrenado) se ejecuta en el **motor nativo** de este directorio, construido sobre [GGML](https://github.com/ggml-org/ggml) y con una API nativa en C++, en macOS, iOS, Android, Linux y Windows. Los pesos de los expertos se cargan desde el almacenamiento bajo demanda en un conjunto fijo de ranuras, de modo que un modelo más grande que el presupuesto de memoria del dispositivo se ejecuta sin cargar todos sus pesos. En un Mac con Apple M5 decodifica a unos 25 tokens por segundo, con un pico de huella física de unos 3,4 GiB; las cifras de cada plataforma están en [Benchmarking](docs/benchmarking.md#reference-figures).
- **Edge0 145B** y **Edge0 120B** se ejecutan en **edge0-strata**, un fork de [Strata](https://github.com/Niko1221/Strata) en `edge0-strata/` del repositorio Edge0, en una GPU NVIDIA con Linux o Windows; consulta [Edge0 145B](docs/edge0-145b.md) y [Edge0 120B](docs/edge0-120b.md).

> Esta página es una traducción del [README](README.md) en inglés; los documentos enlazados están en inglés.

## Plataformas

| Plataforma | Edge0 35B (motor nativo) | Edge0 145B, Edge0 120B (edge0-strata) |
| --- | --- | --- |
| macOS (Apple silicon) | Metal, CPU | — |
| iOS | Metal | — |
| Android | CPU | — |
| Linux | CPU | GPU NVIDIA (CUDA 13) |
| Windows | CPU, Vulkan | GPU NVIDIA (CUDA 13) |

Mediciones: [Benchmarking](docs/benchmarking.md#reference-figures) para Edge0 35B y la página del modelo Edge0 145B y Edge0 120B (`docs/models/edge0-145b.md` en el repositorio Edge0). En el motor nativo, el prefill y el decode pueden ejecutarse en backends distintos del mismo dispositivo, por ejemplo prefill en Metal y decode en CPU. El 35B ofrece dos modos de enrutamiento: `predicted` (por defecto), en el que un pre-router permite al motor cargar los expertos del siguiente token mientras se calcula el actual, y `actual`, el cálculo original del modelo. Consulta [Routing modes](docs/benchmarking.md#routing-modes).

## Inicio rápido

Ejecuta los comandos desde este directorio, `edge0-infer/` en un clon del [repositorio Edge0](https://github.com/Edge0-AI/Edge0):

```sh
cmake --preset mac-metal && cmake --build --preset mac-metal
build/mac-metal/bin/edge0-infer info
```

Después descarga y convierte los recursos del modelo, y ejecuta una instrucción, un chat o un servidor local compatible con OpenAI:

```sh
build/mac-metal/bin/edge0-infer run   --model "$MOE_ASSETS_35B" --device metal --user "Preséntate en una frase."
build/mac-metal/bin/edge0-infer chat  --model "$MOE_ASSETS_35B" --device metal
build/mac-metal/bin/edge0-infer serve --model "$MOE_ASSETS_35B" --device metal
```

[Getting started](docs/getting-started.md) recorre cada paso, incluidas la descarga y la conversión de los recursos y las compilaciones para Android, Windows e iOS.

Para Edge0 145B y Edge0 120B, compila edge0-strata y ejecuta `edge0-infer serve --model <carpeta del modelo>`; los pasos están en [Getting started](docs/getting-started.md#edge0-145b-and-edge0-120b-edge0-strata).

## Documentación

- [Getting started](docs/getting-started.md): compilar, obtener los recursos del modelo, ejecutar, chatear y servir, en cada plataforma
- [Edge0 145B](docs/edge0-145b.md): ejecutar este modelo en edge0-strata
- [Edge0 120B](docs/edge0-120b.md): ejecutar este modelo en edge0-strata
- [Configuration reference](docs/configuration.md): ajustes de runtime, carga, modelo y generación, sus valores por defecto y las opciones de la CLI
- [Native API](docs/native-api.md): uso del motor dentro del proceso: ciclo de vida, streaming, varios turnos, concurrencia y cancelación
- [OpenAI-compatible local server](docs/openai-compatible-server.md): endpoints, campos de la solicitud y límites
- [Architecture](docs/architecture.md): las dos capas, cómo está organizado el motor nativo y cómo fluye una solicitud por él
- [Known limitations](docs/limitations.md): comportamientos a tener en cuenta
- [Benchmarking](docs/benchmarking.md): cómo medir, ajustes recomendados y cifras de referencia
- [Testing](docs/testing.md): niveles de prueba, qué necesita cada uno y cómo ejecutarlos
- [Model asset tools](tools/assets/README.md): el directorio de recursos, las comprobaciones, los fixtures de prueba pequeños y cómo añadir un modelo
- [Integration examples](examples/README.md): el ejemplo de la API nativa y el cliente del SDK de OpenAI
- [Chat applications](examples/README.md#chat-applications): ejemplos para escritorio (macOS y Windows), Android e iOS
- [Measurement tools](tests/perf/README.md): referencia para desarrolladores de los scripts de medición y los informes

## API nativa

La API nativa corresponde al motor nativo. Las aplicaciones enlazan el target de CMake `edge0::edge0` e incluyen las cabeceras de [`include/edge0/`](include/edge0/), que son la única fuente de las firmas; los tipos y funciones están en el espacio de nombres `edge0`. [`examples/native_chat.cpp`](examples/native_chat.cpp) muestra el ciclo de vida completo: abrir el runtime, cargar un modelo, abrir una sesión, generar dos turnos en streaming y liberar todo. La [guía de la API nativa](docs/native-api.md) explica el ciclo de vida, el streaming, el uso con varios turnos, la concurrencia, la cancelación y los errores.

## Licencia y código de terceros

Este directorio se distribuye bajo la Apache License 2.0 del repositorio que lo contiene. El código de terceros conserva su propia licencia:

- GGML, incluido sin modificaciones en `third_party/ggml/` (MIT, [`third_party/GGML_LICENSE`](third_party/GGML_LICENSE));
- cpp-httplib (MIT, [`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)) y nlohmann/json (MIT, [`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)), usados solo por la CLI y el servidor;
- el código del tokenizador adaptado de llama.cpp en `src/app/text/` (MIT, [`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)).

## Comentarios y contribuciones

Informa de problemas y propón cambios mediante issues y pull requests en el repositorio que contiene este directorio. El directorio se desarrolla en un repositorio aparte y se sincroniza aquí: un cambio aceptado se aplica primero allí y llega aquí con la siguiente sincronización.
