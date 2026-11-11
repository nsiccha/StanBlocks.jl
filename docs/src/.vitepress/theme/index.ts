import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'virtual:mathjax-styles.css'
import ScientificFigure from '../components/ScientificFigure.vue'
import './style.css'
import './docstrings.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ScientificFigure', ScientificFigure)
  },
} satisfies Theme
