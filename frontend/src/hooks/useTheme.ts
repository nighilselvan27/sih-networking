import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark' | 'system'

const KEY = 'unindr.theme'

function read(): Theme {
  try {
    const stored = localStorage.getItem(KEY)
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      return stored
    }
  } catch {
    /* storage unavailable */
  }
  return 'system'
}

function apply(theme: Theme) {
  const root = document.documentElement

  if (theme === 'system') root.removeAttribute('data-theme')
  else root.setAttribute('data-theme', theme)
}

export function useTheme(): [Theme, (next: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(read)

  useEffect(() => {
    apply(theme)
  }, [theme])

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next)

    try {
      localStorage.setItem(KEY, next)
    } catch {
      /* storage unavailable; the choice applies for this session only */
    }
  }, [])

  return [theme, setTheme]
}

/** Applied before React mounts so the first paint is already correct. */
export function initTheme(): void {
  apply(read())
}
