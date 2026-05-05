// .vitepress/theme/index.ts
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import 'virtual:mathjax-styles.css';

import { 
  NolebaseEnhancedReadabilitiesMenu, 
  NolebaseEnhancedReadabilitiesScreenMenu, 
} from '@nolebase/vitepress-plugin-enhanced-readabilities/client'

import VersionPicker from "@/VersionPicker.vue"
import AuthorBadge from '@/AuthorBadge.vue'
import Authors from '@/Authors.vue'
import Banner from '@/Banner.vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

import '@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css'
import './style.css' // You could setup your own, or else a default will be copied.
import './docstrings.css' // You could setup your own, or else a default will be copied.

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(Banner),
      'nav-bar-content-after': () => [
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      'nav-screen-content-after': () => h(NolebaseEnhancedReadabilitiesScreenMenu),
    })
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
    app.component('AuthorBadge', AuthorBadge)
    app.component('Authors', Authors)

    // VitePress strips raw `<script>` blocks from markdown (they get
    // hoisted as Vue SFC scripts, not executed at the embed location).
    // So the `data-hx-base` → `hx-get` rewrite has to live here, where
    // it actually runs in the SPA lifecycle. We wire two triggers:
    //
    //   1. Route change (SPA navigation): re-scan the new page.
    //   2. Initial mount: poll briefly until the embed exists in the
    //      DOM (Vue renders it after enhanceApp runs).
    //
    // Each scan finds `[data-hx-base]:not([hx-get])` elements, sets
    // `hx-get` to `<deploy-base>/<data-hx-base>gallery`, then calls
    // `htmx.process(el)` to fire its `hx-trigger="load"`.
    if (typeof window !== 'undefined' && router) {
      const processEmbeds = () => {
        // @ts-ignore - htmx loaded via head <script>; no types.
        const htmx = (window as any).htmx;
        // siteData.value.base is `/` in dev, `/StanBlocks.jl/dev/` in prod.
        const base = (siteData?.value?.base || '/').replace(/\/$/, '');
        document.querySelectorAll('[data-hx-base]:not([hx-get])').forEach((el) => {
          const tail = el.getAttribute('data-hx-base') || '';
          const url = base + '/' + tail + 'gallery';
          el.setAttribute('hx-get', url);
          // Explicit ajax — htmx.process(el) doesn't reliably re-fire
          // `hx-trigger="load"` on an already-scanned element.
          if (htmx) htmx.ajax('GET', url, { target: el as HTMLElement, swap: 'innerHTML' });
        });
      };
      router.onAfterRouteChanged = processEmbeds;
      // Initial-mount: enhanceApp runs before Vue's first render, so
      // poll briefly until the embed appears (or htmx loads).
      let tries = 0;
      const tick = () => {
        if (tries++ > 200) return; // ~10s ceiling
        const pending = document.querySelector('[data-hx-base]:not([hx-get])');
        // @ts-ignore
        if (!pending || !(window as any).htmx) {
          setTimeout(tick, 50);
          return;
        }
        processEmbeds();
      };
      if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', tick);
        } else {
          tick();
        }
      }

      // Embedded HTMXO fragments contain root-absolute links like
      // `<a href="/sandbox_view/foo">` that point at the StanBlocks
      // server's own paths. Without rewriting they resolve against
      // VitePress's origin and 404. Rewrite root-absolute hrefs and
      // hx-* URLs inside `.htmxo-embed` containers to go through the
      // configured prefix (default `/live-sb`). Override per-page via
      // `<meta name="htmxo-embed-prefix" content="…">`.
      const rewritePrefix = (
        document.querySelector('meta[name="htmxo-embed-prefix"]')?.getAttribute('content')
        ?? '/live-sb'
      );
      const rewriteRootRefs = (root: HTMLElement) => {
        const fix = (el: Element, attr: string) => {
          const v = el.getAttribute(attr);
          if (v && v.startsWith('/') && !v.startsWith('//') && !v.startsWith(rewritePrefix)) {
            el.setAttribute(attr, rewritePrefix + v);
          }
        };
        root.querySelectorAll('[href]').forEach((el) => fix(el, 'href'));
        root.querySelectorAll('[hx-get]').forEach((el) => fix(el, 'hx-get'));
        root.querySelectorAll('[hx-post]').forEach((el) => fix(el, 'hx-post'));
        root.querySelectorAll('[hx-put]').forEach((el) => fix(el, 'hx-put'));
        root.querySelectorAll('[hx-patch]').forEach((el) => fix(el, 'hx-patch'));
        root.querySelectorAll('[hx-delete]').forEach((el) => fix(el, 'hx-delete'));
      };
      document.body.addEventListener('htmx:afterSwap', (e: any) => {
        const tgt = e?.detail?.target as HTMLElement | undefined;
        if (!tgt) return;
        const embed = tgt.closest('.htmxo-embed');
        if (!embed) return;
        rewriteRootRefs(embed as HTMLElement);
      });
    }
  }
}
export default Theme