import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { SystemProvider } from './state/SystemContext'
import { Alerts } from './pages/Alerts'
import { Benchmarks } from './pages/Benchmarks'
import { Flows } from './pages/Flows'
import { LiveMonitor } from './pages/LiveMonitor'
import { Models } from './pages/Models'
import { Overview } from './pages/Overview'
import { Settings } from './pages/Settings'
import { Traffic } from './pages/Traffic'

/*
 * HashRouter is deliberate: the built console is served as static files
 * from /console by the API, which has no rewrite rule for deep links. A
 * hash route works there and when the dist directory is opened directly.
 */
export function App() {
  return (
    <SystemProvider>
      <HashRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Overview />} />
            <Route path="/live" element={<LiveMonitor />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/traffic" element={<Traffic />} />
            <Route path="/flows" element={<Flows />} />
            <Route path="/models" element={<Models />} />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </SystemProvider>
  )
}
