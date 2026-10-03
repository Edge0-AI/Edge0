# Edge0Phone

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

Chat embarqué pour iPhone. L'application peut exécuter Edge0 8B, Edge0 35B, ou les deux, et seul le modèle que vous sélectionnez est chargé. Les poids des modèles ne sont pas dans ce dossier.

## Démarrage rapide

Vous avez besoin d'un Mac Apple Silicon, de Xcode et d'un iPhone sous iOS 17 ou ultérieur avec le mode développeur activé. Python 3.10 ou ultérieur n'est nécessaire que pour télécharger les poids et pour convertir le 35B. Si Xcode signale un compilateur Metal manquant, exécutez `xcodebuild -downloadComponent MetalToolchain`.

Le répertoire `Models/` est copié dans l'application au moment de la compilation. Placez-y un modèle avant de compiler si vous voulez que ce bouton soit sélectionnable. Un répertoire `Models/` vide compile quand même ; le modèle manquant affiche **Not included**.

### Télécharger un modèle

```sh
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install --upgrade pip huggingface_hub numpy
```

Le 8B peut être empaqueté tel que téléchargé. Il pèse environ 5 GB, l'application installée a donc à peu près cette taille.

```sh
.venv-tools/bin/hf download Edge0/Edge0-8B-A1B-preview \
  --local-dir Models/Edge0-8B-A1B-preview
```

Le 35B est un changement de disposition, pas une modification des poids. Le checkpoint publié stocke chaque expert sous forme de neuf lectures dispersées dans un fichier de 19 GB. Sur un iPhone, c'est trop lent, donc les outils ci-dessous réécrivent chaque expert en une seule lecture séquentielle et écrivent un fichier par couche. Conservez le téléchargement d'origine hors de `Models/`. Seul `Models/repacked/` doit contenir ce que l'application doit embarquer. Une compilation 35B pèse environ 20 GB.

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

`Models/repacked/` doit contenir `resident.safetensors`, `tokenizer.bin`, ainsi que `experts-L00.bin` à `experts-L39.bin`. Vous pouvez empaqueter l'un ou l'autre modèle, ou les deux.

Checkpoints :

- [Edge0/Edge0-8B-A1B-preview](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
- [Edge0/Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)

### Compilation et installation

`com.example.edge0phone` dans le projet est un identifiant de bundle fictif. Avant de compiler ou de signer, remplacez-le par un identifiant de bundle unique que vous contrôlez. Dans Xcode, ouvrez `Edge0PhoneProbe.xcodeproj`, sélectionnez la cible **Edge0Phone**, puis modifiez **Signing & Capabilities → Bundle Identifier**. Choisissez votre équipe Apple Developer, connectez l'iPhone et lancez (Run).

Pour définir l'équipe depuis un fichier, copiez `Config/Local.xcconfig.example` vers `Config/Local.xcconfig` et renseignez `DEVELOPMENT_TEAM`.

La même compilation depuis la ligne de commande :

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

Les poids se trouvent à l'intérieur de ce `.app`. Il n'y a pas de copie séparée vers le téléphone.

### Tests

`scripts/test.sh` compile la bibliothèque MLX Metal avec Xcode, puis exécute les tests unitaires. La première exécution résout le paquet MLX Swift.

```sh
scripts/test.sh
EDGE0_TEST_GPU=1 scripts/test.sh
```

### Utiliser l'application

Lancez l'application et choisissez **Edge0 8B** ou **Edge0 35B**. Un modèle qui n'était pas dans `Models/` au moment de la compilation affiche **Not included** et ne peut pas être sélectionné. Saisissez une question ou touchez l'une des suggestions. La ligne sous une réponse affiche le nombre de tokens, le temps avant le premier token, le prefill, le decode et la mémoire de pointe. Changer de modèle décharge celui qui est chargé. New chat efface la conversation.

## Performances

| Modèle | Appareil | OS | Prefill | TTFT | Decode |
| --- | --- | --- | --- | --- | --- |
| 8B | iPhone 16 Pro | iOS 26.6.2 | 7.2 tok/s | 3.6 s | 10.9 tok/s |
| 35B | iPhone 16 Pro | iOS 26.6.2 | 4.9 tok/s | 2.1 s | 6.4 tok/s |

## Détails techniques

L'application est en SwiftUI. `Edge0MLX` exécute le modèle. `Edge0Core` lit le checkpoint, le tokenizer et l'index de routage. Les calculs mathématiques passent par MLX Swift sur Metal. La cible de déploiement est iOS 17.

Edge0 8B comporte 24 couches, alternant Kimi Delta Attention et attention latente multi-têtes, Top-8, INT4. Le prefill du prompt exécute une couche entière à la fois, par blocs de 2048 tokens au plus. Le premier token est glouton (greedy). Les tokens suivants utilisent une température de 0,7, top-k 64, top-p 0,95 et une pénalité de répétition de 1,1.

Edge0 35B comporte 40 couches, 256 experts par couche, Top-4. Les poids résidents et les fichiers d'experts par couche sont lus séparément. Le cache de préfixes est stocké dans Application Support, car le bundle de l'application est en lecture seule. Les deux modèles ne sont jamais chargés en même temps.
