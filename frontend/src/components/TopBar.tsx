import { Menu, Monitor, Moon, PanelLeft, Sun } from 'lucide-react'
import { useLocation } from 'react-router-dom'
import { useStream } from '@/hooks/useStream'
import { useTheme } from '@/hooks/useTheme'
import type { Theme } from '@/hooks/useTheme'
import { findRoute } from '@/routes'
import { captureDescription, useSystem } from '@/state/SystemContext'
import { Status } from './Indicators'
import type { Tone } from './Indicators'

/*
 * Application header.
 *
 * Left: where you are. Right: whether the system is working. Two system
 * indicators and a theme control — not a toolbar of buttons.
 */

const THEME_ORDER: Theme[] = ['system', 'light', 'dark']

const THEME_ICON = {
  system: Monitor,
  light: Sun,
  dark: Moon,
} as const

function streamIndicator(
  status: ReturnType<typeof useStream>['status'],
  paused: boolean,
): { tone: Tone; label: string } {
  if (paused) return { tone: 'warn', label: 'Stream paused' }

  switch (status) {
    case 'open':
      return { tone: 'accent', label: 'Stream connected' }
    case 'connecting':
      return { tone: 'muted', label: 'Stream connecting' }
    case 'error':
      return { tone: 'danger', label: 'Stream disconnected' }
    default:
      return { tone: 'muted', label: 'Stream idle' }
  }
}

export function TopBar({
  onToggleSidebar,
  onOpenMobileNav,
}: {
  onToggleSidebar: () => void
  onOpenMobileNav: () => void
}) {
  const { pathname } = useLocation()
  const route = findRoute(pathname)

  const { stats, error } = useSystem()
  const stream = useStream()
  const [theme, setTheme] = useTheme()

  const capture = captureDescription(stats, error)
  const streamState = streamIndicator(stream.status, stream.paused)

  const ThemeIcon = THEME_ICON[theme]

  const cycleTheme = () => {
    const index = THEME_ORDER.indexOf(theme)
    setTheme(THEME_ORDER[(index + 1) % THEME_ORDER.length] ?? 'system')
  }

  return (
    <header
      className="flex h-header shrink-0 items-center gap-3 border-b
                 border-border bg-surface px-4"
    >
      <button
        type="button"
        onClick={onOpenMobileNav}
        className="btn-quiet -ml-1.5 rounded p-1.5 md:hidden"
        aria-label="Open navigation"
      >
        <Menu className="h-4 w-4" aria-hidden />
      </button>

      <button
        type="button"
        onClick={onToggleSidebar}
        className="btn-quiet -ml-1.5 hidden rounded p-1.5 md:inline-flex"
        aria-label="Toggle sidebar"
      >
        <PanelLeft className="h-4 w-4" aria-hidden />
      </button>

      <div className="min-w-0 leading-tight">
        <h1 className="truncate text-sm font-medium text-text">
          {route?.label ?? 'UniNDR'}
        </h1>
        {route?.description ? (
          <p className="truncate text-2xs text-text-3">{route.description}</p>
        ) : null}
      </div>

      <div className="ml-auto flex items-center gap-4">
        <span className="hidden text-xs lg:inline-flex" title={capture.detail}>
          <Status
            tone={capture.tone}
            label={capture.label}
            muted
            pulse={capture.tone === 'ok'}
          />
        </span>

        <span className="hidden text-xs sm:inline-flex">
          <Status tone={streamState.tone} label={streamState.label} muted />
        </span>

        <button
          type="button"
          onClick={cycleTheme}
          className="btn-quiet rounded p-1.5"
          aria-label={`Theme: ${theme}. Switch theme.`}
          title={`Theme: ${theme}`}
        >
          <ThemeIcon className="h-4 w-4" aria-hidden />
        </button>
      </div>
    </header>
  )
}
