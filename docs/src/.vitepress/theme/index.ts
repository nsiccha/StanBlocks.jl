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

// Synced from HTMXObjects/assets/vitepress/htmxo-embed.ts by
// `HTMXObjects.vitepress_theme_install` in make.jl. Don't edit in place
// — edit the upstream and re-run make.jl.
import { setupHtmxoEmbed } from './htmxo-embed'
import { setupAtlasComparisons } from './atlas-comparison'

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
    // HTMXObjects embed wiring: data-hx-base resolution + SPA route
    // re-process + .htmxo-embed link rewriting. Sb defaults the proxy
    // prefix to `/live-sb` (matches the Vite proxy in config.mts and
    // committed recordings under public/live-sb/).
    setupHtmxoEmbed(router, { proxyPrefix: '/live-sb' });
    setupAtlasComparisons();
  }
}
export default Theme
