# edge0-infer

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

Un motor de inferencia en el dispositivo para modelos de lenguaje de mezcla de expertos (MoE), construido sobre [GGML](https://github.com/ggml-org/ggml). Un solo motor y una sola API nativa en C++ ejecutan el mismo modelo en macOS, iOS, Android y Windows. Los pesos de los expertos se cargan desde el almacenamiento bajo demanda en un conjunto fijo de ranuras, de modo que un modelo más grande que el presupuesto de memoria del dispositivo se ejecuta sin cargar todos sus pesos.

El primer modelo compatible es **Edge0 35B** (Qwen3.6-35B-A3B con un LoRA y un pre-router entrenado). En un Mac con Apple M5 decodifica a unos 25 tokens por segundo, con un pico de huella física de unos 3,4 GiB; las cifras de cada plataforma están en [Benchmarking](docs/benchmarking.md#reference-figures).

> Esta página es una traducción del [README](README.md) en inglés; los documentos enlazados están en inglés.

## Plataformas

| Plataforma | Backend | Verificado en |
| --- | --- | --- |
| macOS (Apple silicon) | Metal, CPU | Apple M5, 16 GiB |
| iOS | Metal | iPhone 17 Pro Max |
| Android | CPU | OnePlus PKR110 (Snapdragon 8 Elite) |
| Windows (x86-64) | CPU, Vulkan | Intel Core i5-12400, NVIDIA GeForce RTX 4070 SUPER |

El prefill y el decode pueden ejecutarse en backends distintos del mismo dispositivo, por ejemplo prefill en Metal y decode en CPU. El 35B ofrece dos modos de enrutamiento: `predicted` (por defecto), en el que un pre-router permite al motor cargar los expertos del siguiente token mientras se calcula el actual, y `actual`, el cálculo original del modelo. Consulta [Routing modes](docs/benchmarking.md#routing-modes).

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

## Documentación

- [Getting started](docs/getting-started.md): compilar, obtener los recursos del modelo, ejecutar, chatear y servir, en cada plataforma
- [Configuration reference](docs/configuration.md): ajustes de runtime, carga, modelo y generación, sus valores por defecto y las opciones de la CLI
- [Native API](docs/native-api.md): uso del motor dentro del proceso: ciclo de vida, streaming, varios turnos, concurrencia y cancelación
- [OpenAI-compatible local server](docs/openai-compatible-server.md): endpoints, campos de la solicitud y límites
- [Architecture](docs/architecture.md): cómo está organizado el motor y cómo fluye una solicitud por él
- [Known limitations](docs/limitations.md): comportamientos a tener en cuenta
- [Benchmarking](docs/benchmarking.md): cómo medir, ajustes recomendados y cifras de referencia
- [Testing](docs/testing.md): niveles de prueba, qué necesita cada uno y cómo ejecutarlos
- [Model asset tools](tools/assets/README.md): el directorio de recursos, las comprobaciones, los fixtures de prueba pequeños y cómo añadir un modelo
- [Integration examples](examples/README.md): el ejemplo de la API nativa y el cliente del SDK de OpenAI
- [Chat applications](examples/README.md#chat-applications): ejemplos para escritorio (macOS y Windows), Android e iOS
- [Measurement tools](tests/perf/README.md): referencia para desarrolladores de los scripts de medición y los informes

## API nativa

Las aplicaciones enlazan el target de CMake `edge0::edge0` e incluyen las cabeceras de [`include/edge0/`](include/edge0/), que son la única fuente de las firmas; los tipos y funciones están en el espacio de nombres `edge0`. [`examples/native_chat.cpp`](examples/native_chat.cpp) muestra el ciclo de vida completo: abrir el runtime, cargar un modelo, abrir una sesión, generar dos turnos en streaming y liberar todo. La [guía de la API nativa](docs/native-api.md) explica el ciclo de vida, el streaming, el uso con varios turnos, la concurrencia, la cancelación y los errores.

## Licencia y código de terceros

Este directorio se distribuye bajo la Apache License 2.0 del repositorio que lo contiene. El código de terceros conserva su propia licencia:

- GGML, incluido sin modificaciones en `third_party/ggml/` (MIT, [`third_party/GGML_LICENSE`](third_party/GGML_LICENSE));
- cpp-httplib (MIT, [`third_party/cpp-httplib/LICENSE`](third_party/cpp-httplib/LICENSE)) y nlohmann/json (MIT, [`third_party/nlohmann-json/LICENSE.MIT`](third_party/nlohmann-json/LICENSE.MIT)), usados solo por la CLI y el servidor;
- el código del tokenizador adaptado de llama.cpp en `src/app/text/` (MIT, [`src/app/text/LICENSE.llama.cpp`](src/app/text/LICENSE.llama.cpp)).

## Comentarios y contribuciones

Informa de problemas y propón cambios mediante issues y pull requests en el repositorio que contiene este directorio. El directorio se desarrolla en un repositorio aparte y se sincroniza aquí: un cambio aceptado se aplica primero allí y llega aquí con la siguiente sincronización.
