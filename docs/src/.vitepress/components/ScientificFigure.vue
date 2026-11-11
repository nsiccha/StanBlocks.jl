<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{ src: string; title: string }>()
const frame = ref<HTMLIFrameElement>()
let themeObserver: MutationObserver | undefined
let sizeObserver: ResizeObserver | undefined

function syncTheme() {
  const doc = frame.value?.contentDocument
  if (!doc) return
  const dark = document.documentElement.classList.contains('dark')
  doc.documentElement.classList.toggle('dark', dark)
  doc.documentElement.style.colorScheme = dark ? 'dark' : 'light'
  doc.body.style.color = dark ? '#e5e5e5' : '#0b0b0b'
  doc.body.style.background = dark ? '#1b1b1f' : '#ffffff'
  doc.body.style.fontFamily = getComputedStyle(document.body).fontFamily
}
function loaded() {
  syncTheme()
  sizeObserver?.disconnect()
  const doc = frame.value?.contentDocument
  if (!doc || !frame.value) return
  const resize = () => {
    if (frame.value) frame.value.style.height = `${doc.body.scrollHeight + 24}px`
  }
  sizeObserver = new ResizeObserver(resize)
  sizeObserver.observe(doc.body)
  resize()
}
onMounted(() => {
  themeObserver = new MutationObserver(syncTheme)
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
})
onBeforeUnmount(() => {
  themeObserver?.disconnect()
  sizeObserver?.disconnect()
})
</script>

<template>
  <iframe ref="frame" :src="withBase(src)" :title="title" class="scientific-figure" @load="loaded" />
</template>
