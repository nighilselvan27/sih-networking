import clsx from 'clsx'
import { useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { findRoute } from '@/routes'
import { ErrorBoundary } from './ErrorBoundary'
import { StaleBanner } from './StaleBanner'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

/*
 * Application shell: sidebar, header, scrolling content.
 *
 * Desktop-first. The sidebar collapses to icons on medium screens and
 * becomes an overlay drawer below that, because a security console is used
 * on a desktop but must not break on a phone.
 */

const COLLAPSE_KEY = 'unindr.sidebarCollapsed'

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === '1'
  } catch {
    return false
  }
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState<boolean>(readCollapsed)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const { pathname } = useLocation()

  // Keep the document title in step with the route, so browser tabs and
  // history entries are distinguishable.
  useEffect(() => {
    const route = findRoute(pathname)
    document.title = route ? `${route.label} · UniNDR` : 'UniNDR'
  }, [pathname])

  useEffect(() => {
    setMobileNavOpen(false)

    // The content area scrolls, not the window, so a route change would
    // otherwise land the operator part-way down the new page.
    document.getElementById('main')?.scrollTo({ top: 0 })
  }, [pathname])

  useEffect(() => {
    if (!mobileNavOpen) return

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setMobileNavOpen(false)
    }

    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [mobileNavOpen])

  const toggleCollapsed = () => {
    setCollapsed((current) => {
      const next = !current
      try {
        localStorage.setItem(COLLAPSE_KEY, next ? '1' : '0')
      } catch {
        /* storage unavailable */
      }
      return next
    })
  }

  return (
    <div className="flex h-full">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3
                   focus:top-3 focus:z-50 focus:rounded focus:border
                   focus:border-border focus:bg-surface focus:px-3
                   focus:py-1.5 focus:text-xs"
      >
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside
        className={clsx(
          'hidden shrink-0 border-r border-border md:block',
          collapsed ? 'w-sidebar-collapsed' : 'w-sidebar',
        )}
      >
        <Sidebar collapsed={collapsed} />
      </aside>

      {/* Mobile sidebar */}
      {mobileNavOpen ? (
        <div className="fixed inset-0 z-40 md:hidden">
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setMobileNavOpen(false)}
            className="absolute inset-0 bg-scrim animate-overlay-in
                       motion-reduce:animate-none"
          />
          <div
            className="absolute inset-y-0 left-0 w-sidebar border-r
                       border-border shadow-overlay"
          >
            <Sidebar
              collapsed={false}
              onNavigate={() => setMobileNavOpen(false)}
            />
          </div>
        </div>
      ) : null}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          onToggleSidebar={toggleCollapsed}
          onOpenMobileNav={() => setMobileNavOpen(true)}
        />

        <main
          id="main"
          className="scroll-y flex-1 overflow-y-auto"
          tabIndex={-1}
        >
          <div className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6 sm:py-6">
            <StaleBanner />
            <ErrorBoundary resetKey={pathname}>
              <Outlet />
            </ErrorBoundary>
          </div>
        </main>
      </div>
    </div>
  )
}
