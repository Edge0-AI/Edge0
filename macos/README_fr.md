# edge0

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | Français

Inférence embarquée pour Apple Silicon. L'application communique avec un serveur local compatible OpenAI qui exécute les modèles MoE Edge0 (8B et 35B) avec des poids d'experts en streaming, SSD offload et Metal.

Nécessite **macOS 14+** sur **Apple Silicon (M3 ou ultérieur)**. Toutes les commandes ci-dessous s'exécutent depuis la racine de ce répertoire. MLX v0.30.6 est déjà présent sous `third_party/mlx`.

## Démarrage rapide

### Prérequis

- Xcode Command Line Tools (compilateur Metal)
- CMake 3.24+
- Rust 1.88 (`rustup` choisira la version depuis `rust-toolchain.toml`)
- Node 22 ou ultérieur (yarn 1.22 est utilisé s'il est déjà installé ; sinon `check-prereqs` l'active via Corepack)

`make app` exécute cette vérification. Pour vérifier d'abord :

```bash
bash scripts/check-prereqs.sh
```

### Compilation et packaging

```bash
make app
open dist/edge0-0.1.0-arm64.app
```

`make app` installe les dépendances JavaScript, compile `edge0`, `edge0d` et `edge0-engine`, compile le moteur natif, puis génère `dist/edge0-<version>-arm64.app`. Il ne lance ni l'application ni le service local. Le `.app` est signé ad hoc.

### Premier lancement

1. Ouvrez le `.app`. Le service local (`edge0d` sur `127.0.0.1:8000`) démarre avec lui.
2. Sur la page Models, téléchargez un modèle depuis Hugging Face (stocké sous `~/.edge0`), chargez-le, puis discutez.
3. Chat : Entrée envoie. Les réponses arrivent en streaming au format Markdown, y compris les formules mathématiques `$inline$` et `$$block$$`.
4. API HTTP (le loopback ne nécessite pas de token) :

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "edge0-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "stream": false
  }'
```

L'accès hors loopback nécessite `Authorization: Bearer <token>` depuis la page Service.

### Développement

Pour itérer sans packaging :

```bash
make setup
cargo build -p edge0d -p edge0-engine
make engine
cd app && yarn tauri dev
```

Cela lance une fenêtre de développement qui peut démarrer le service local. `make engine` compile la bibliothèque native MLX/Metal.

### Utilisation de l'application

| Écran | Rôle |
| --- | --- |
| Chat | Chat en streaming, prompts prédéfinis, style de génération (Precise / Balanced / Creative), Markdown + formules mathématiques |
| Models | Téléchargement, chargement/déchargement, maintien actif (keep-alive) |
| Service | Démarrer/arrêter le démon, liaison LAN, LaunchAgent, token API, vérification de l'environnement, journal des requêtes |
| Settings | Thème, taille de police, prompt système optionnel |
| Menu bar | Status, Open window, Restart service, Quit (arrête le démon que cette application a démarré) |

Les données résident dans `~/.edge0` (modèles, base de données des sessions, journaux). La CLI se trouve dans `Contents/Resources/bin/edge0` à l'intérieur du `.app`. `edge0 uninstall` arrête le service et tout LaunchAgent ; ajoutez `--purge` pour supprimer les données utilisateur après confirmation.

## Performances

Mesurées sur **MacBook Air, Apple M3**.

| Modèle | Decode | Prefill | Mémoire résidente |
| --- | --- | --- | --- |
| Edge0-35B-A3B | 10–12 tok/s | ~70–130 tok/s | 4 GB |
| Edge0-8B-A1B | 18–20 tok/s | ~300–570 tok/s | 1.7 GB |

## Détails techniques

```
edge0.app (Tauri 2 + React 19)
    HTTP 127.0.0.1:8000  →  edge0d (OpenAI-compatible + /v1/edge0/*)
    Unix socket frames   →  edge0-engine
    dlopen               →  libedge0_engine_native.dylib + libmlx.dylib
```

- **UI :** Tauri 2, React 19, TanStack Router, Tailwind, Vercel AI SDK, streamdown + `@streamdown/math` (KaTeX).
- **Démon :** Rust. `/v1/chat/completions` (SSE). Téléchargements depuis Hugging Face par défaut.
- **Moteur :** C++/Metal sur mlx 0.30.6. Experts int4 en streaming (tranches mmap + pile LRU à chaud + SSD offload), préchargement prerouter, LoRA non fusionné, KV cache de préfixe. NAX GEMM est désactivé par défaut (`MLX_METAL_NO_NAX`) car mlx 0.30.6 est numériquement incorrect pour edge0-8b sur M5.

`EDGE0_HOME` a pour valeur par défaut `~/.edge0`.
