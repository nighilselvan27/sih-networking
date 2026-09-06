# UniNDR console

Operator console for the CTU-13 hybrid IDS. React + TypeScript + Vite +
Tailwind.

```bash
npm install
npm run dev        # http://localhost:5173, proxies the API on :8000
npm run build      # emits dist/, served by the API at /console
npm run typecheck
```

Set `IDS_API_URL` to proxy somewhere other than `http://127.0.0.1:8000`, or
change the API base URL at runtime on the Settings page.

## Layout

```
src/
  lib/         api.ts (typed client) · types.ts (backend contract) ·
               format.ts · stream.ts (SSE with backoff reconnect)
  hooks/       usePolling · useStream · useTheme · useChartColors
  state/       SystemContext — one shared poller for stats and health
  components/  AppShell · Sidebar · TopBar · FlowTable · FlowDetail ·
               Charts · Drawer · Metric · Indicators · Controls ·
               Section · States · BenchmarkTable · ModelPipeline ·
               DemoControls · ErrorBoundary · StaleBanner
  pages/       Overview · LiveMonitor · Alerts · Traffic · Flows ·
               Models · Benchmarks · Settings
  styles/      tokens.css (the palette) · index.css
```

## Rules this codebase follows

**Nothing is invented.** `src/lib/types.ts` mirrors the backend responses
exactly. If a field is not declared there, the backend does not return it and
the UI must not display it. In particular the detector is binary — there is no
attack family, no per-feature attribution and no live accuracy figure, so none
is shown.

**Stale is never shown as live.** When the API stops answering, the figures
already on screen are dimmed and `StaleBanner` says how old they are. Silently
leaving a healthy-looking number on screen is the worst failure a monitoring
console can have.

**Colour is state, not decoration.** Four semantic colours: red malicious /
critical, amber elevated, green healthy, blue active. Severity uses three
distinct readings — red, amber, and plain secondary text — so the column can be
scanned.

**Tokens only.** `tailwind.config.ts` replaces the default palette entirely, so
a component cannot reach for a colour that is not in `tokens.css` and light and
dark stay in step by construction. Note that Tailwind cannot apply an opacity
modifier to a hex custom property (`bg-text/10` computes to transparent) —
overlays use the `--scrim` token, which carries its own alpha.

**Borders over shadows.** One shadow token, used by the drawer and popovers.
Sections are separated by rules, not boxed in cards.

**Monospace is for identifiers.** Addresses, ports, flow ids, scores,
thresholds, timestamps and protocol strings. Nothing else.

**Motion communicates state.** New table rows fade in over 120 ms; status dots
transition colour. Charts do not animate on update — a line that moves and
fades at once is harder to read. `prefers-reduced-motion` disables all of it.

## Live stream

One `EventSource` for the whole app, shared through `useSyncExternalStore`
(`lib/stream.ts`). Two invariants keep it correct and fast:

- `state` is only replaced inside `emit`, which then notifies. A silent
  mutation makes `useSyncExternalStore` re-render in a loop.
- Arriving flows are coalesced and applied every 250 ms, and table rows are
  memoised. Capture can deliver flows faster than the browser can lay out a
  table; one render per event lets the stream rate dictate the frame rate.
