import { useEffect, useState } from 'react'

/*
 * Recharts writes colours as SVG presentation attributes, where `var(--x)`
 * is not valid. So the tokens are resolved to concrete values here and
 * re-resolved whenever the theme changes — which keeps charts on the same
 * palette as the rest of the interface without hard-coding a single hex.
 */

const TOKENS = [
  'series-1',
  'series-2',
  'series-3',
  'grid',
  'text-2',
  'text-3',
  'border',
  'surface',
  'accent',
  'danger',
  'warn',
  'ok',
] as const

export type ChartColors = Record<(typeof TOKENS)[number], string>

function read(): ChartColors {
  const styles = getComputedStyle(document.documentElement)

  return Object.fromEntries(
    TOKENS.map((token) => [
      token,
      styles.getPropertyValue(`--${token}`).trim() || '#737373',
    ]),
  ) as ChartColors
}

export function useChartColors(): ChartColors {
  const [colors, setColors] = useState<ChartColors>(read)

  useEffect(() => {
    const refresh = () => setColors(read())

    // Explicit theme switch.
    const observer = new MutationObserver(refresh)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })

    // System preference change while set to "system".
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', refresh)

    return () => {
      observer.disconnect()
      media.removeEventListener('change', refresh)
    }
  }, [])

  return colors
}
