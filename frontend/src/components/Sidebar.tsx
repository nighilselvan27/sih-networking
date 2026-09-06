import clsx from 'clsx'
import { NavLink } from 'react-router-dom'
import { NAV } from '@/routes'
import { captureDescription, useSystem } from '@/state/SystemContext'
import { StatusDot } from './Indicators'

/*
 * Sidebar.
 *
 * Narrow and quiet: a small product mark, three labelled groups, and one
 * honest status line at the foot. No large status card, no product
 * illustration.
 */

function Mark() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="h-4 w-4 shrink-0 text-accent"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {/* Three nodes and a monitored link: a network under inspection. */}
      <circle cx="3" cy="3.5" r="1.6" />
      <circle cx="13" cy="3.5" r="1.6" />
      <circle cx="8" cy="12.5" r="1.6" />
      <path d="M4.3 4.7 6.9 11M11.7 4.7 9.1 11M4.6 3.5h6.8" />
    </svg>
  )
}

export function Sidebar({
  collapsed,
  onNavigate,
}: {
  collapsed: boolean
  onNavigate?: () => void
}) {
  const { stats, error } = useSystem()
  const capture = captureDescription(stats, error)

  return (
    <div className="flex h-full flex-col bg-surface">
      <div
        className={clsx(
          'flex h-header shrink-0 items-center border-b border-border',
          collapsed ? 'justify-center px-2' : 'gap-2.5 px-4',
        )}
      >
        <Mark />
        {!collapsed && (
          <div className="min-w-0 leading-tight">
            <div className="text-sm font-medium tracking-[-0.01em] text-text">
              UniNDR
            </div>
            <div className="text-2xs text-text-3">Network Detection</div>
          </div>
        )}
      </div>

      <nav
        className={clsx(
          'scroll-y flex-1 overflow-y-auto py-3',
          collapsed ? 'px-2' : 'px-2.5',
        )}
        aria-label="Main"
      >
        {NAV.map((section) => (
          <div key={section.label} className="mb-4 last:mb-0">
            {!collapsed && (
              <div className="label-field px-2 pb-1.5">{section.label}</div>
            )}
            <ul className="space-y-px">
              {section.items.map((item) => (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    end={item.path === '/'}
                    onClick={onNavigate}
                    title={collapsed ? item.label : undefined}
                    className={({ isActive }) =>
                      clsx(
                        'flex items-center rounded text-xs transition-colors',
                        collapsed
                          ? 'justify-center px-0 py-2'
                          : 'gap-2.5 px-2 py-1.5',
                        isActive
                          ? 'bg-surface-3 font-medium text-text'
                          : 'text-text-2 hover:bg-surface-2 hover:text-text',
                      )
                    }
                  >
                    <item.icon
                      className="h-4 w-4 shrink-0"
                      strokeWidth={1.5}
                      aria-hidden
                    />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div
        className={clsx(
          'shrink-0 border-t border-border py-2.5',
          collapsed ? 'flex justify-center px-2' : 'px-4',
        )}
        title={collapsed ? `${capture.label} — ${capture.detail}` : undefined}
      >
        {collapsed ? (
          <StatusDot tone={capture.tone} />
        ) : (
          <div className="flex items-center gap-1.5 text-2xs">
            <StatusDot tone={capture.tone} />
            <span className="truncate text-text-2">{capture.label}</span>
          </div>
        )}
      </div>
    </div>
  )
}
