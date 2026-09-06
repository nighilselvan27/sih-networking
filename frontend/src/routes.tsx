import {
  Activity,
  BarChart3,
  Bell,
  Boxes,
  Gauge,
  LayoutGrid,
  Network,
  Settings as SettingsIcon,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/*
 * Navigation is the single source of truth for page identity: the sidebar,
 * the header title and the document title all read from here.
 */

export interface RouteDef {
  path: string
  label: string
  /** Shown under the title in the application header. */
  description: string
  icon: LucideIcon
}

export interface NavSection {
  label: string
  items: RouteDef[]
}

export const NAV: NavSection[] = [
  {
    label: 'Monitor',
    items: [
      {
        path: '/',
        label: 'Overview',
        description: 'Real-time network security monitoring',
        icon: LayoutGrid,
      },
      {
        path: '/live',
        label: 'Live Monitor',
        description: 'Real-time unidirectional flow detection',
        icon: Activity,
      },
      {
        path: '/alerts',
        label: 'Alerts',
        description: 'Detected anomalies and security events',
        icon: Bell,
      },
    ],
  },
  {
    label: 'Analysis',
    items: [
      {
        path: '/traffic',
        label: 'Traffic',
        description: 'Volume, protocol mix and top talkers',
        icon: BarChart3,
      },
      {
        path: '/flows',
        label: 'Flows',
        description: 'Scored flow history and feature detail',
        icon: Network,
      },
      {
        path: '/models',
        label: 'Models',
        description: 'Detection architecture and operating thresholds',
        icon: Boxes,
      },
    ],
  },
  {
    label: 'System',
    items: [
      {
        path: '/benchmarks',
        label: 'Benchmarks',
        description: 'Recorded evaluation results on CTU-13',
        icon: Gauge,
      },
      {
        path: '/settings',
        label: 'Settings',
        description: 'Connection, display and diagnostics',
        icon: SettingsIcon,
      },
    ],
  },
]

export const ROUTES: RouteDef[] = NAV.flatMap((section) => section.items)

export function findRoute(pathname: string): RouteDef | undefined {
  return (
    ROUTES.find((route) => route.path === pathname) ??
    ROUTES.find(
      (route) => route.path !== '/' && pathname.startsWith(route.path),
    )
  )
}
