# Edge0Phone

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | Español | [Français](README_fr.md)

Chat en el dispositivo para iPhone. La aplicación puede ejecutar Edge0 8B, Edge0 35B o ambos, y solo se carga el modelo que selecciones. Los pesos del modelo no están en esta carpeta.

## Inicio rápido

Necesitas un Mac con Apple Silicon, Xcode y un iPhone con iOS 17 o posterior y el modo de desarrollador activado. Python 3.10 o posterior solo se necesita para descargar los pesos y para convertir el 35B. Si Xcode indica que falta el compilador de Metal, ejecuta `xcodebuild -downloadComponent MetalToolchain`.

El directorio `Models/` se copia dentro de la aplicación al compilar. Coloca ahí un modelo antes de compilar si quieres que ese botón sea seleccionable. Un directorio `Models/` vacío compila igualmente; el modelo ausente muestra **No incluido**.

### Descargar un modelo

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip huggingface_hub numpy
```

El 8B se puede empaquetar tal como se descarga. Ocupa unos 5 GB, así que la aplicación instalada tiene aproximadamente ese tamaño.

```sh
.venv-tools/bin/hf download Edge0/Edge0-8B-A1B-preview \
  --local-dir Models/Edge0-8B-A1B-preview
```

El 35B es un cambio de disposición (layout), no un cambio en los pesos. El checkpoint publicado almacena cada experto como nueve lecturas dispersas a lo largo de un archivo de 19 GB. En un iPhone eso es demasiado lento, así que las herramientas de abajo reescriben cada experto en una sola lectura secuencial y escriben un archivo por capa. Conserva la descarga original fuera de `Models/`. Solo `Models/repacked/` es lo que debe contener la aplicación. Una compilación del 35B ocupa unos 20 GB.

```sh
mkdir -p checkpoints
.venv-tools/bin/hf download Edge0/Edge0-35B-A3B-preview \
  --local-dir checkpoints/Edge0-35B-A3B-preview

SOURCE=checkpoints/Edge0-35B-A3B-preview
OUTPUT=Models/repacked

.venv-tools/bin/python tools/repack_experts.py pack "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/repack_experts.py resident "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/repack_experts.py verify "$SOURCE" "$OUTPUT"
.venv-tools/bin/python tools/convert_tokenizer.py "$SOURCE" "$OUTPUT/tokenizer.bin"
.venv-tools/bin/python tools/convert_pregate.py \
  "$SOURCE/prerouter_edge0_35b.safetensors" \
  "$OUTPUT/prerouter-stacked.safetensors"
cp "$SOURCE/lora_edge0_35b.safetensors" "$OUTPUT/"
```

`Models/repacked/` debe contener `resident.safetensors`, `tokenizer.bin` y desde `experts-L00.bin` hasta `experts-L39.bin`. Puedes empaquetar cualquiera de los dos modelos, o ambos.

Checkpoints:

- [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
- [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)

### Compilación e instalación

`com.example.edge0phone` en el proyecto es un identificador de bundle de ejemplo. Antes de compilar o firmar, reemplázalo por un identificador de bundle único que controles. En Xcode, abre `Edge0PhoneProbe.xcodeproj`, selecciona el target **Edge0Phone** y cambia **Signing & Capabilities → Bundle Identifier**. Elige tu Apple Developer Team, conecta el iPhone y pulsa Run.

Para configurar el equipo desde un archivo, copia `Config/Local.xcconfig.example` a `Config/Local.xcconfig` y completa `DEVELOPMENT_TEAM`.

La misma compilación desde la línea de comandos:

```sh
DEVICE_ID=<device id from `xcrun devicectl list devices`>
TEAM_ID=<your Apple Developer Team ID>
BUNDLE_ID=com.yourname.edge0phone # replace with your own unique bundle identifier

xcodebuild -project Edge0PhoneProbe.xcodeproj \
  -scheme Edge0Phone \
  -destination "id=$DEVICE_ID" \
  -derivedDataPath .build-device \
  -allowProvisioningUpdates \
  -skipPackagePluginValidation \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM="$TEAM_ID" \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
  build

xcrun devicectl device install app \
  --device "$DEVICE_ID" \
  .build-device/Build/Products/Debug-iphoneos/Edge0Phone.app
```

Los pesos van dentro de ese `.app`. No hay una copia independiente al teléfono.

### Pruebas

`scripts/test.sh` compila la biblioteca MLX Metal con Xcode y después ejecuta las pruebas unitarias. La primera ejecución resuelve el paquete MLX Swift.

```sh
scripts/test.sh
EDGE0_TEST_GPU=1 scripts/test.sh
```

### Uso de la aplicación

Abre la aplicación y elige **Edge0 8B** o **Edge0 35B**. Un modelo que no estaba en `Models/` al compilar muestra **No incluido** y no se puede seleccionar. Escribe una pregunta o toca una de las sugerencias. La línea bajo una respuesta muestra el recuento de tokens, el tiempo hasta el primer token, prefill, decode y memoria pico. Al cambiar de modelo se libera el que estaba cargado. Nuevo chat borra la conversación.

## Rendimiento

| Modelo | Dispositivo | SO | Prefill | TTFT | Decode |
| --- | --- | --- | --- | --- | --- |
| 8B | iPhone 16 Pro | iOS 26.6.2 | 7.2 tok/s | 3.6 s | 10.9 tok/s |
| 35B | iPhone 16 Pro | iOS 26.6.2 | 4.9 tok/s | 2.1 s | 6.4 tok/s |

## Detalles técnicos

La aplicación es SwiftUI. `Edge0MLX` ejecuta el modelo. `Edge0Core` lee el checkpoint, el tokenizer y el índice de enrutamiento. Las fórmulas matemáticas pasan por MLX Swift sobre Metal. El objetivo de despliegue es iOS 17.

Edge0 8B tiene 24 capas, alternando Kimi Delta Attention y atención latente multicabeza, Top-8, INT4. El prefill del prompt ejecuta una capa completa cada vez, en fragmentos de hasta 2048 tokens. El primer token es greedy. Los tokens siguientes usan temperatura 0.7, top-k 64, top-p 0.95 y una penalización de repetición de 1.1.

Edge0 35B tiene 40 capas, 256 expertos por capa, Top-4. Los pesos residentes y los archivos de expertos por capa se leen por separado. La caché de prefijos se guarda en Application Support, porque el bundle de la aplicación es de solo lectura. Los dos modelos nunca se cargan al mismo tiempo.
