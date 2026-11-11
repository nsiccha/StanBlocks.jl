# Vendored Vega / Vega-Lite / Vega-Embed builds

Exact-pinned browser builds served by `vega_head(; source=:vendor|:inline)`
and `to_html(...; source=:inline)` so AoV plots render with no CDN access.
`:cdn` (the default) serves these same builds from jsDelivr with
subresource-integrity hashes; the three modes render identical output.

| File              | Package    | Version | sha256                                                           |
|-------------------|------------|---------|------------------------------------------------------------------|
| vega.min.js       | vega       | 5.33.1  | 463f3db6a40b20e9747b4ed38f37ed0add508838f9141b1cf8366784b07b30c8 |
| vega-lite.min.js  | vega-lite  | 5.23.0  | 58c27358e26f2d319cf62f45bc17a4c8362f08645001df2ec8d341eee4097c7f |
| vega-embed.min.js | vega-embed | 6.29.0  | 12d02acfbe3ec59ef9a37dd4822a2e04e2961b5bbb671bbe661d2221715b99da |

Provenance: `https://cdn.jsdelivr.net/npm/<pkg>@<version>/build/<file>`,
byte-identical to the bare `https://cdn.jsdelivr.net/npm/<pkg>@<version>` URL
the major-only tags used to resolve to (verified 2026-09-28). Bytes are stored
pristine, including the trailing `sourceMappingURL` comments (a missing `.map`
fails silently in devtools; no runtime effect).

Update recipe (bump the trio):

1. Download the three `build/*.min.js` files at the new versions and confirm
   each is byte-identical to its bare-URL form.
2. Replace the files here; update the table above.
3. Update `VEGA_VERSION` / `VEGALITE_VERSION` / `VEGA_EMBED_VERSION` and the
   `*_SRI` (sha384) constants in `src/js_runtime.jl`.
4. Update the sha256 lock in `test/items.jl` ("vendored vega trio" testitem).
5. Re-check the closing tags `source=:inline` rewrites ("inline script bodies
   never spell a closing tag" testitem): every `</script`, `</style`,
   `</head`, `</body` and `</html` in the new builds must sit in a string,
   template literal, regex or comment (never `String.raw`), where `<\/` reads
   the same. 6.29.0's vega-embed has three, in its source-view page template.

## Number and time locales

`locales/d3-format/{de-DE,en-US}.json` and
`locales/d3-time-format/{de-DE,en-US}.json` are the unmodified locale files from
[d3-format v3.1.0](https://github.com/d3/d3-format/tree/v3.1.0/locale) and
[d3-time-format v4.1.0](https://github.com/d3/d3-time-format/tree/v4.1.0/locale).
Each directory includes that release's ISC license. `locale_config` reads
these bundled files; it does not fetch data at runtime. The byte hashes in
`test/locales.jl` pin their provenance. To update, fetch the locale files and
licenses from the new tags, then update the hashes and check rendered formats.

## Inline elements and closing tags

Inlined bytes are written through `_inline_text` (`src/AlgebraOfVega.jl`):
`</` before `script`, `style`, `head`, `body` or `html` becomes `<\/`. The
first two would end the element early; the document-level three are what
text-substituting proxies key on (nginx `sub_filter '</head>' …` with
`sub_filter_once off`, injected analytics/livereload snippets), and such a
filter rewrites every occurrence, including one inside a script. The vendored
builds stay pristine; only their inline copy differs, at those sites.

## AoV's own runtime and stylesheet

`aov-runtime.js` (the `window.AoV.*` browser runtime) and `aov.css` (the head
stylesheet) are AoV's own sources, not vendored third-party builds: edit them
here directly. They are the single source of both forms `vega_head` emits —
inlined by default (`vega_runtime()` returns the runtime inline), and loaded
by URL under `vega_head(source=:vendor, runtime=:linked)`, so a page carries a
few hundred bytes of tags instead of ~96 KB the browser can cache.

Every `source=:vendor` URL — the trio and, when linked, these two — is
`<base>/<file>?v=<first 16 hex of the file's sha256>`, so the bytes behind one
URL never change and an app may serve this directory with
`Cache-Control: public, max-age=31536000, immutable`. Both files are inlined
by default, byte for byte, so neither may spell a closing `script` / `style` /
`head` / `body` / `html` tag — write `<\/` in string literals (asserted in
`test/items.jl`, "vega script source modes"); they are not hash-locked.
