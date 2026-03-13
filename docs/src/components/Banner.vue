<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { frontmatter, site } = useData()

// Priority: per-page frontmatter > global themeConfig > dev-branch auto-warning
const msg = computed(() => {
  if (frontmatter.value.warning) return frontmatter.value.warning
  if (site.value.themeConfig.warning) return site.value.themeConfig.warning
  if (site.value.base.includes('/dev/')) return 'You are viewing the dev branch. This branch may include code written with Claude Code with less human supervision. Only human-approved code is merged into main.'
  return null
})
</script>

<template>
  <div v-if="msg" class="warning-banner">{{ msg }}</div>
</template>

<style scoped>
.warning-banner {
  background: var(--vp-c-warning-soft, #fff3cd);
  color: var(--vp-c-warning-1, #856404);
  text-align: center;
  padding: 0.5rem 1rem;
  font-size: 1.05em;
  font-weight: 700;
  position: sticky;
  bottom: 0;
  z-index: 100;
}
</style>
