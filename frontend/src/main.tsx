import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { initTheme } from './hooks/useTheme'
import './styles/index.css'

// Applied before the first paint so the interface never flashes the wrong
// theme on load.
initTheme()

const container = document.getElementById('root')

if (!container) {
  throw new Error('Root element not found')
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
