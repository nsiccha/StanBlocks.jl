import { defineConfig } from 'vitepress'
import { mathjaxPlugin } from './mathjax-plugin'
import { stanLanguage } from './stan-language'
const mathjax = mathjaxPlugin()

export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'Probabilistic model composition in Julia, compiled to Stan.',
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    config(md) { mathjax.markdownConfig(md) },
    theme: { light: 'github-light', dark: 'github-dark' },
    languages: [stanLanguage],
  },
  vite: { plugins: [mathjax.vitePlugin] },
  themeConfig: {
    outline: [2, 3],
    nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }],
  },
})
