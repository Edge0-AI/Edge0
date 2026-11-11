# Is Strata safe and private?

This page says what Strata downloads, where from, when it uses the internet, what stays on your PC and how the downloads
are checked. Everything here comes from reading the code of this version. Each section names the files, so you or anyone
you trust can check it. Where Strata does less than you might hope, the page says so.

## At a glance

- **Your chats, prompts, files, pictures and models stay on your PC.** They leave only when you choose it: you add an
  outside model service, you open the server to other devices or to the internet, or the agent searches the web or opens a page
  (see [What the agent can do](#what-the-coding-agent-can-do-on-your-pc)).
- **There is no telemetry.** Strata sends no usage data, no crash reports and no analytics, and has no account to sign in to.
- **Starting and running a model never downloads anything**, except that `strata launch` updates an installed DeepSeek
  engine package by itself when the signed feed has a newer one (below). Other downloads and updates are steps that you start.
- **The server listens on your PC only.** Reaching it from another device needs an API key.
- **Once a model is downloaded, Strata runs without internet.** Starting, loading, serving and chatting need no
  connection. Only things you ask for use it: the agent's web search and web pages, an outside model service, the tunnel,
  and installs and updates (a first model install also fetches llama.cpp's preparation scripts), plus the DeepSeek engine
  check of `strata launch` (without a connection it gives up after at most 10 s and the launch goes on).
- **Downloads are checked.** Updates come from a signed feed. Every model file has a fixed SHA-256 and size, and an install
  stops if the pinned version of a file is gone (the details are [below](#how-downloads-are-checked)).
- **In its default mode the coding agent asks before it changes anything**, and once per chat before its first web search or web page.
- **The code is open.** Anyone can read it, and security problems go to [SECURITY.md](../SECURITY.md).

## Does Strata phone home?

No. A search of the app, the Terminal, the engine server and the installers found no analytics library, no crash
reporter and no code that reports usage anywhere. The app's privacy notes say the same
(`apps/host/src/components/PrivacyNotes.tsx`: "Strata sends no usage telemetry").

Some things sound similar and stay on your PC:

- The Monitor, `/metrics` and `/v1/telemetry` show hardware readings (temperature, speed, memory) to you. They are served
  from your own PC, and they need your API key when you set one.
- When the app window crashes (Linux) or the app fails to start (Windows, Mac), a crash log is written to a folder on your disk
  (on Windows, `%LOCALAPPDATA%\Strata\logs`). Nothing sends it. **Save logs**
  (Settings, Engine, Diagnostics) puts a text file in your Downloads folder, and you decide whether to attach it to a bug report.
- The coding agent is a separate component (the DeepSeek Harness, MIT). It has its own telemetry switch, and Strata starts it
  with that switch off (`apps/host/server/agent/harness-runtime.mjs`).

Like any download, each connection below shows the other side your IP address and the file you asked for.

## When Strata uses the internet

| What | Where it goes | When |
| --- | --- | --- |
| Model files, the picture reader, the speed-up layer (MTP) | huggingface.co, or the mirror in `HF_ENDPOINT` if you set one | When you install a model |
| The same files from ModelScope, for PCs that cannot reach Hugging Face | modelscope.cn | Only if you ask for it (`--source modelscope`). Strata never switches to it by itself |
| Engine, Terminal and Control updates | The signed feed on GitHub Releases of Niko1221/Strata | When you install an engine, press **Update engine** or **Check for updates**, or run `strata update`; the DeepSeek engine also on `strata launch` (below). Opening **Settings, Engine** does not connect: it shows the result of your last check |
| App updates | `latest.json` on the same GitHub Releases, through Tauri's updater | When you press **Check for updates** or **Update Strata** |
| llama.cpp source code (a fixed commit) | github.com/ggml-org/llama.cpp | During a model install, if it is not on your PC yet. Strata uses only its model-preparation scripts |
| Web search and web pages | DuckDuckGo, with Bing as the second source; or Exa, Perplexity or DeepSeek if you save a key for them. Any page the agent is asked to open goes to that page's site | When the agent uses its search or "open web pages" tool. In Manual mode it asks you first, once per chat |
| Outside model services | The service you add (OpenAI, Anthropic, OpenRouter, a server on your network and others) | Only for chats with that model, and only after you add its key |
| Cloudflare tunnel | Cloudflare, through the `cloudflared` program | Only while you have the tunnel on, and Strata makes an API key for it unless you pass `--no-api-key` (then it warns that anyone who finds the address can use your model) |
| Tailscale, local network | Not Strata's traffic. Strata only reads Tailscale's status and shows the address | Only if you choose that access in `strata launch` |
| An image link in an API request | The address in the link | Only when a program sends your server a request with an `http(s)` image link. `--offline` refuses it |
| Remote tool servers (MCP) | The address you put in the run config | Only when you add one |
| A website you open in the app's preview pane | That website | Only when you type an address into **Open a website** |
| Share to X or Reddit (Monitor screenshot) | Your browser opens x.com or reddit.com with a caption and a link to the Strata page | Only when you click it. The screenshot is saved to your disk, not uploaded |

That is the full list found in the code. Some details:

- **Nothing checks for updates in the background.** The app does not check when it starts, and not when you open Settings,
  Engine either: that screen shows the result of your last check, or nothing if you have not checked. The checks in the table
  run when you press a button or run a command. With no connection, the check says so and nothing else happens. The Terminal fetches an engine by itself only when none is
  installed, and `strata launch` asks first. One exception: `strata launch` with a DeepSeek model first checks whether the
  signed feed can be reached (at most 10 s; without a connection it says nothing and goes on), then reads it, and when it has
  a newer DeepSeek engine package for this PC, fetches and checks it, starts it once (`--version`), switches to it and removes
  the older one. On a failure the installed one keeps running. Not when the launch is given `--dsv41-package` or
  `STRATA_DSV41_EXE` is set (`strata_cli/launch/runner.py`, `upgrade_dsv41_engine`).
- **Starting and serving never download.** `strata serve` and the app's load button fetch nothing. The app sets
  `STRATA_PACKAGED=1` and `STRATA_NO_ACQUIRE=1` for its parts (`apps/desktop/src-tauri/src/main.rs`), and the downloaders
  refuse to run under them (`strata_control/acquisition.py`, `apps/host/server/acquisition.mjs`). That is why the app
  never downloads `cloudflared` for you while it runs.
- **The engine itself has no network code.** The part of Strata that runs the model (`src/`, `include/`, `sycl/`, `metal/`)
  contains no networking. The app's own pages load no fonts, scripts or trackers from the internet.
- **Building from source** reaches more places. Python from python.org or winget (`START-HERE.bat`), a standalone Python
  from GitHub on a Mac (`setup.sh`), Python packages from PyPI (`requirements.txt`, pinned versions), NVIDIA's CUDA libraries
  from PyPI and, on Ubuntu and Debian, NVIDIA's apt key and packages, AMD's ROCm wheels, the Vulkan SDK from LunarG, and
  llama.cpp and stable-diffusion.cpp source from GitHub. These run only while you run the setup.

## What stays on your PC

- **Chats, prompts, attached files and pictures** are handled by the engine on your PC. The app keeps chats in its own
  storage on your disk. Models are in the folder you chose.
- **API keys** you save for outside services are kept in a file that only your user account can open. They are not
  encrypted. (`apps/host/server/model-providers.mjs`)
- **The server listens on `127.0.0.1`**, this PC only. The app fixes that address (`STRATA_BIND=127.0.0.1`), and the engine
  server's default is the same (`serve/server.py`, the `--host` option).
- **Sharing needs a key.** The app's "Share with other devices" and the Cloudflare tunnel will not start without an API
  key, and the shared port answers only `/v1/models`, `/v1/chat/completions` and `/v1/messages`
  (`apps/host/server/local-share.mjs`, `cloudflare.mjs`). `strata launch` makes a key for you when you pick anything wider than
  this computer. The bare engine server (`serve/server.py`) only prints a warning if you open it to your network without a key,
  so set one yourself in that case.
- **When you use an outside model**, the conversation (and the instructions of skills you switched on) go to that service.
  The app says so in its privacy notes. The service's own rules then apply.
- **Programs you connect** (Claude Code, Codex, and so on) are separate programs with their own network behaviour. Strata
  does not control what they send.

The server also has browser protections (a host check, an origin check, no CORS by default). They are in
[SECURITY.md](../SECURITY.md#what-the-server-exposes).

## How downloads are checked

**Updates for the engine, the Terminal and Control.**

1. The release publishes `strata-feed.json` and `strata-feed.json.sig`, signed with an Ed25519 key. The matching public key is
   inside every installed Strata (`strata_control/update_keys.json`). The private key is kept offline by the owner.
2. Before it uses the feed, your PC checks the signature, that the feed has not expired and that it is not older than the
   last feed it accepted (so an old feed cannot be replayed).
3. Each engine package is checked against its SHA-256 and its signed manifest, then every file in it against its own hash,
   then it is checked to fit your PC.
4. The new engine is test-started before Strata switches to it. If anything fails, you keep the engine you had.

Updates never replace your models or saved tuning. The full description is in [packaging/UPDATES.md](../packaging/UPDATES.md).

**The app.** Its updates are signed with a separate key (Tauri's minisign), and the public half is in the app
(`apps/desktop/src-tauri/tauri.conf.json`). The updater refuses an installer whose signature does not match. The Terminal
checks the same signature when you run `strata update --app`.

**Model files.** Every model file that `setup.py` downloads has a fixed SHA-256 and size, written in the code, so the check
needs no network: the main Qwen3.8-Flash-Next files (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), Swift 1.5, the Coder, the Unsloth sizes,
the Gyro files, Qwen3.6, the NVFP4 packs, the picture reader, the image model files and the speed-up layer (MTP, tensor by
tensor). A file that does not match is deleted and the install stops. A file whose size on the server is not the pinned
size stops it before the download starts.

- If the pinned version of a file is gone from Hugging Face, the install stops and says to update Strata. It does not take
  the repository's current file instead. A tester can set `STRATA_ALLOW_UNPINNED=1` to take the current file anyway, without
  the check. It is off by default, and setup says so when it is set.
- Starting a model does not check its files, and running setup again for a model you have does not download or hash them
  again. A file installed before these hashes existed is kept as it is.
- Files from ModelScope are checked against the same fixed SHA-256. ModelScope serves a repository's current files, so a file
  it has changed fails the check. A file without a pin is checked against the hash ModelScope publishes, and setup warns
  when it publishes none.
- The app installs these models by running the same setup, so it has the same checks. The app's host also keeps a short
  starter list of its own (`apps/host/server/real-model-catalog.mjs`). It is for display only. The host refuses to download
  from it, because its files have no hash, and says that models come from the Models screen.
  (`setup.py`: `REPO_PINS`, `download`, `verify_sha256`; `tools/mtp_fetch.py`; `apps/host/server/image-models.mjs`.)

**Engine in the source route.** `setup.py` checks the engine zip against the SHA-256 that GitHub's API publishes for that
release file. If GitHub cannot say (offline, rate limit), setup warns and installs anyway. `STRATA_SKIP_SHA256=1` switches the
check off, and it is never on by default.

**The one-line install** (`curl ... | sh`, `irm ... | iex`) checks the Terminal archive's SHA-256 and the signed feed with the
Terminal it just unpacked. The checksum and the key come from the same release as the archive, so the first install trusts
HTTPS and the `Niko1221/Strata` release itself. What that means, and how to read the script first, is in
[SECURITY.md](../SECURITY.md#how-far-to-trust-curl--sh-and-irm--iex). Later updates are checked against the key already on
your PC.

**Code signing.** The Windows installer is not code-signed yet, so Windows shows "Windows protected your PC". The signing and
Apple notarisation of the Mac builds is set up in `packaging/mac/` and runs on the maintainer's Mac. A Mac build that has not
been notarised shows macOS's "damaged" warning ([MAC.md](MAC.md)).

**What the checks do not cover.** They catch a damaged or swapped download and a feed nobody signed with the release key. They
cannot protect you if someone who can publish a release (or who holds the signing key) publishes something harmful.

## What the coding agent can do on your PC

The app's chats run through an agent that can read files, run commands, edit files, search the web and use tools you
connect. What it may do without asking depends on the **Mode** menu (`apps/host/server/agent/README.md`).

| Mode | What happens |
| --- | --- |
| **Manual** (the default) | Anything that changes something asks first, with one **Approve** or **Deny** card, before it runs. |
| **Plan** | Nothing can change. A request to change something is refused, and no card appears. Tools you connect are refused too. |
| **Bypass** | Full access, no questions. |

In **Manual**:

- **These ask:** writing or editing a file, any command that is not on a short list of look-only commands, and every tool
  you connect. In the browser tool, opening a page, clicking and typing ask.
- **These do not ask:** reading files, listing and searching folders, and look-only commands such as `ls`, `cat` and
  `git status`. Each of those commands runs unasked only with options that just look: `sort -o FILE`, `rg --pre PROGRAM`,
  `tail -f`, `date -s` and `git grep -O` ask, as does any option the check does not know. A command with a redirect, a
  chain (`;`, `&&`), a variable or anything else the check does not recognise asks.
- **Web search and opening web pages ask once per chat.** The first one in a chat shows a card with the search words and the
  service they go to, or with the address of the page. **Approve** covers the rest of that chat. **Deny** tells the assistant
  you did not allow the web, and it carries on without it. Plan and Bypass do not ask. The model picks the search words and
  the address, so words from your conversation can end up in a search or a web address, which is why the first one asks.
  In Settings, Tools you can switch the web tools off, or turn on **Let the assistant search the web without asking**, which
  is off by default. A scheduled task in Manual mode cannot answer the card, so its web call is refused unless that switch is
  on (`strata-gate.mjs`).
- **The chat folder is only where the agent starts.** Manual mode does not confine it there. What you
  approve can touch anything your user account can touch, so read the card before you press Approve.
- **Your keys stay out of its commands.** The agent's shell does not get environment variables whose names contain KEY, TOKEN,
  SECRET, PASSWORD, CREDENTIAL or AUTH.
- **Scheduled tasks** run in Plan mode by default. With nobody at the screen, a request for approval is answered "no".
- **Stop** ends a chat's run. Closing a tab does not.

The agent runs on a loopback address with a random token, so other programs on the network cannot reach it.

## Open source and reporting a problem

Strata is MIT-licensed and the whole code is at [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata). Read it, build
it and compare. Strata has not had an outside security audit yet.

To report a security problem, use GitHub's private reporting (the repository's **Security** tab), as described in
[SECURITY.md](../SECURITY.md). Ordinary questions and ideas can go to an
[issue](https://github.com/Niko1221/Strata/issues).

## Where to check this in the code

| Topic | Files |
| --- | --- |
| Model downloads and pins | `setup.py`, `tools/mtp_fetch.py`, `tools/image/qimage.py`, `apps/host/server/models.mjs`, `image-models.mjs` |
| Update feed, signatures | `strata_control/feed.py`, `signing.py`, `transfer.py`, `update_keys.json`, `strata_cli/update_cmd.py` |
| App updater | `apps/desktop/src-tauri/src/updater.rs`, `apps/desktop/src-tauri/tauri.conf.json` |
| Downloads refused while running | `strata_control/acquisition.py`, `apps/host/server/acquisition.mjs` |
| One-line installers | `packaging/install.sh`, `packaging/install.ps1` |
| Web tools and the agent's approvals | `apps/host/server/agent/plugins/strata-gate.mjs`, `agent/profile.mjs`, `web-search-ddg.mjs`, `web-search-bing.mjs`, `web-search-keyed.mjs` |
| Outside model services | `apps/host/server/model-providers.mjs` |
| Sharing, tunnel | `apps/host/server/local-share.mjs`, `cloudflare.mjs`, `strata_cli/launch/access.py` |
