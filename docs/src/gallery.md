---
aside: false
outline: false
---

<style>
/* Break the gallery embed out of VitePress's narrow content column. */
.VPDoc:has(.htmxo-embed) > .container > .content { max-width: none !important; }
.VPDoc:has(.htmxo-embed) .content-container { max-width: none !important; }

/* Override the inline 2-column gallery grid with a docs-friendly
 * auto-fit layout: as many cards per row as fit at ≥520px each, so
 * Stan code blocks have horizontal room. Drops to 1 col on phones. */
.htmxo-embed [style*="grid-template-columns"][style*="repeat"] {
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)) !important;
    gap: 0.75rem !important;
}

.htmxo-embed article {
    overflow: visible !important;
    min-width: 0;
}
.htmxo-embed article pre {
    overflow-x: auto;
}
</style>

# Sandbox Gallery

The StanBlocks **sandbox** is a directory of one-file SLIC snippets at
[`web/sandbox/`](https://github.com/nsiccha/StanBlocks.jl/tree/main/web/sandbox)
— each `.jl` is an `@slic` block (or a `@deffun` + `@slic` pair) that
exercises a specific feature: closures, kwargs, ragged vectors, ODE
solvers, etc. Each snippet's transpiled Stan output and `stanc` status
is cached alongside the source.

::: tip Live and recorded
- **Local app:** [`http://localhost:8091`](http://localhost:8091) — the
  sandbox lives at `/sandbox` (editable) and the read-only gallery at
  `/gallery`.
- **Refresh the deploy recordings:**
  `curl -sX POST http://localhost:8091/record_gallery`. Triggers
  `HTMXObjects.record!` against the live app and dumps every gallery
  URL into `docs/src/public/live-sb/`. Then
  `git add docs/src/public/live-sb && git commit && git push` — CI
  picks up the recordings as static assets.
:::

## Live preview

The sandbox gallery rendered inline below — fetched via HTMX
(`HX-Request: true` → the StanBlocks server returns a body fragment
that drops into this page). VitePress proxies `/live-sb/*` to the
running web app (`SB_DEV_TARGET=http://localhost:8091` by default) in
dev, and to recordings in production.

<div class="htmxo-embed htmxo-embed-fullwidth" data-hx-base="live-sb/" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading sandbox gallery…</em>
</div>

<script>
// Make the embed URL base-aware. In dev `import.meta.env.BASE_URL` is
// `/`, so `data-hx-base="live-sb/"` becomes `/live-sb/` (matches the
// Vite proxy). In prod the base is `/StanBlocks.jl/dev/`, so it
// becomes `/StanBlocks.jl/dev/live-sb/` (matches the committed
// recordings). Same markdown works in both deploys.
(function () {
  function rewrite() {
    const base = (typeof __DEPLOY_ABSPATH__ !== "undefined" && __DEPLOY_ABSPATH__) || "/";
    document.querySelectorAll("[data-hx-base]").forEach((el) => {
      if (el.hasAttribute("hx-get")) return;
      el.setAttribute("hx-get", base.replace(/\/$/, "") + "/" + el.getAttribute("data-hx-base") + "gallery");
      if (window.htmx) window.htmx.process(el);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", rewrite);
  } else {
    rewrite();
  }
})();
</script>

## Endpoints

- `/gallery` — read-only sandbox gallery (this page's source)
- `/sandbox` — interactive editor (rename / save / delete)
- `/sandbox_view/{id}` — single-snippet detail page
- `/record_gallery` — drives `HTMXObjects.record!` over the gallery URL set
