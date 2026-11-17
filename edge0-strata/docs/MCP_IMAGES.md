# MCP raster images in the web chat

Enable this optional feature with `"mcp": {"image_results": true}` in the run configuration (alongside existing MCP settings). The value must be a JSON boolean (`true` or `false`); strings and numbers are rejected. It is disabled by default; text-only tool output and stream event fields remain unchanged when disabled.

MCP image results are cached by SHA-256 next to the run configuration, in `<config-stem>.tool-media/`. The web chat receives small same-origin references instead of embedding base64 in chat history. PNG, JPEG and WebP are supported; SVG and arbitrary file/resource URIs are not rendered. Each call displays at most four images, each with at most 24 million base64 characters (about 18 MB decoded). If the cache cannot be written, the chat retains the text result and logs the cache error.

`GET /tool-media/<sha256>.<extension>` uses the existing server API authentication, validates the basename, and serves only cached raster files. The web client fetches images with its existing authorization headers, displays them outside collapsed tool details, and provides a blob download link. The cache persists across server restarts; if its files are removed, older image references are replaced with an image-omitted text note when preparing later turns. Disabling the feature or removing MCP configuration does the same, while preserving the tool text and allowing the conversation to continue. Malformed references are still rejected. A global disk-retention policy is not implemented yet.

When the server has a vision encoder, tool images enter its existing image preparation path. Subsequent browser turns send `strata-tool-media:<basename>` references; the authenticated server resolves these only inside its cache. With no vision encoder, images remain visible to the user and the model receives the existing text-only fallback. This adds image transport, not a vision model.

Search and image generation are supplied by configured MCP servers. Strata does not bundle a paid search provider or a ComfyUI installation. Machine-specific adapters and lifecycle hooks are not included in this change.

## Validation without a GPU

```powershell
python -m unittest serve.test_mcp serve.test_mcp_media
node --check serve/web/app.js
git diff --check
```

The transport tests use an existing mock engine and scripted tool response. They verify authenticated binary delivery, invalid media rejection, path traversal refusal, streaming metadata, forwarding to vision preparation and subsequent-turn cache resolution. They do not establish visual understanding or generation quality.

## Browser layout checks

Use a configured image MCP fixture and the mock engine; no GPU is needed for the transport or layout check. Include a long tool name and long text result, and check a running call while Stop is visible. At viewport widths 320, 390, 640, 768 and 1280 pixels, the document must not be wider than the viewport and the preview must fit its chat bubble. Also expand tool details, reload the page, and check that the preview and download link return. A missing cache may remove the picture but must not prevent the next text turn.
