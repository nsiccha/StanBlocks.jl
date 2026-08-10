import './atlas-comparison.css'

let comparisonCounter = 0

type PaneName = 'source' | 'stan'

function makeButton(label: string, className: string) {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = className
  button.textContent = label
  return button
}

function enhanceComparison(root: HTMLElement) {
  if (root.dataset.atlasReady === '1') return

  let source = root.querySelector<HTMLElement>(':scope > [data-atlas-pane="source"]')
  let stan = root.querySelector<HTMLElement>(':scope > [data-atlas-pane="stan"]')
  if (!source || !stan) {
    const sourceCode = root.querySelector<HTMLElement>(':scope > div[class*="language-julia"]')
    const stanCode = root.querySelector<HTMLElement>(':scope > div[class*="language-stan"]')
    if (sourceCode && stanCode) {
      source = document.createElement('div')
      stan = document.createElement('div')
      source.dataset.atlasPane = 'source'
      stan.dataset.atlasPane = 'stan'
      sourceCode.before(source)
      stanCode.before(stan)
      source.appendChild(sourceCode)
      stan.appendChild(stanCode)
    }
  }
  if (!source || !stan) return

  root.dataset.atlasReady = '1'
  const id = `atlas-comparison-${++comparisonCounter}`
  const labels: Record<PaneName, string> = {
    source: root.dataset.sourceLabel || 'StanBlocks',
    stan: root.dataset.stanLabel || 'Generated Stan',
  }
  const panes: Record<PaneName, HTMLElement> = { source, stan }
  let active: PaneName = 'source'

  const toolbar = document.createElement('div')
  toolbar.className = 'atlas-comparison__toolbar'

  const tablist = document.createElement('div')
  tablist.className = 'atlas-comparison__tabs'
  tablist.role = 'tablist'
  tablist.setAttribute('aria-label', 'Choose source or generated code')

  const tabs = {} as Record<PaneName, HTMLButtonElement>
  const selectPane = (name: PaneName, focus = false) => {
    active = name
    for (const paneName of ['source', 'stan'] as PaneName[]) {
      const selected = paneName === active
      panes[paneName].hidden = !selected
      tabs[paneName].classList.toggle('is-active', selected)
      tabs[paneName].setAttribute('aria-selected', String(selected))
      tabs[paneName].tabIndex = selected ? 0 : -1
    }
    if (focus) tabs[name].focus()
  }

  for (const name of ['source', 'stan'] as PaneName[]) {
    const tab = makeButton(labels[name], 'atlas-comparison__tab')
    const tabId = `${id}-${name}-tab`
    const panelId = `${id}-${name}-panel`
    tab.id = tabId
    tab.role = 'tab'
    tab.setAttribute('aria-controls', panelId)
    tab.addEventListener('click', () => selectPane(name))
    tabs[name] = tab

    panes[name].id = panelId
    panes[name].classList.add('atlas-comparison__panel')
    panes[name].role = 'tabpanel'
    panes[name].setAttribute('aria-labelledby', tabId)
    tablist.appendChild(tab)
  }

  tablist.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault()
      selectPane(active === 'source' ? 'stan' : 'source', true)
    } else if (event.key === 'Home') {
      event.preventDefault()
      selectPane('source', true)
    } else if (event.key === 'End') {
      event.preventDefault()
      selectPane('stan', true)
    }
  })

  const expand = makeButton('Compare side by side', 'atlas-comparison__expand')
  expand.setAttribute('aria-haspopup', 'dialog')
  toolbar.append(tablist, expand)
  root.prepend(toolbar)

  const dialog = document.createElement('dialog')
  dialog.className = 'atlas-comparison__dialog'
  dialog.setAttribute('aria-label', 'StanBlocks and generated Stan comparison')

  const frame = document.createElement('div')
  frame.className = 'atlas-comparison__dialog-frame'
  const header = document.createElement('header')
  header.className = 'atlas-comparison__dialog-header'
  const title = document.createElement('strong')
  title.textContent = 'Source and generated program'
  const close = makeButton('Close', 'atlas-comparison__close')
  close.setAttribute('aria-label', 'Close side-by-side comparison')
  close.addEventListener('click', () => dialog.close())
  header.append(title, close)

  const columns = document.createElement('div')
  columns.className = 'atlas-comparison__columns'
  for (const name of ['source', 'stan'] as PaneName[]) {
    const column = document.createElement('section')
    column.className = 'atlas-comparison__column'
    const heading = document.createElement('h3')
    heading.textContent = labels[name]
    const content = panes[name].cloneNode(true) as HTMLElement
    content.hidden = false
    content.removeAttribute('id')
    content.removeAttribute('role')
    content.removeAttribute('aria-labelledby')
    content.removeAttribute('data-atlas-pane')
    content.classList.remove('atlas-comparison__panel')
    content.querySelectorAll('.copy').forEach((button) => button.remove())
    column.append(heading, content)
    columns.appendChild(column)
  }

  frame.append(header, columns)
  dialog.appendChild(frame)
  root.appendChild(dialog)

  expand.addEventListener('click', () => {
    dialog.showModal()
    close.focus()
  })
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close()
  })
  dialog.addEventListener('close', () => expand.focus())

  selectPane('source')
}

function processComparisons(root: ParentNode) {
  const candidates = Array.from(root.querySelectorAll<HTMLElement>('[data-atlas-comparison]'))
  if (root instanceof HTMLElement && root.matches('[data-atlas-comparison]')) {
    candidates.unshift(root)
  }
  candidates.forEach(enhanceComparison)
}

export function setupAtlasComparisons() {
  if (typeof window === 'undefined') return

  processComparisons(document.body)
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLElement) processComparisons(node)
      }
    }
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
