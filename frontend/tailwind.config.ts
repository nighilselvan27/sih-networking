import type { Config } from 'tailwindcss'

/*
 * The scale is intentionally narrow. Every value a component can reach for
 * is listed here, which is what keeps the interface consistent: there is no
 * 32px heading, no 20px radius and no second shadow to accidentally use.
 */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    // Replaces Tailwind's default palette entirely. Components can only
    // use tokens, so light and dark stay in step by construction.
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      bg: 'var(--bg)',
      surface: 'var(--surface)',
      'surface-2': 'var(--surface-2)',
      'surface-3': 'var(--surface-3)',
      text: 'var(--text)',
      'text-2': 'var(--text-2)',
      'text-3': 'var(--text-3)',
      border: 'var(--border)',
      'border-strong': 'var(--border-strong)',
      accent: 'var(--accent)',
      'accent-weak': 'var(--accent-weak)',
      ok: 'var(--ok)',
      'ok-weak': 'var(--ok-weak)',
      warn: 'var(--warn)',
      'warn-weak': 'var(--warn-weak)',
      danger: 'var(--danger)',
      'danger-weak': 'var(--danger-weak)',
      grid: 'var(--grid)',
      scrim: 'var(--scrim)',
    },
    fontFamily: {
      sans: [
        'Inter',
        '-apple-system',
        'BlinkMacSystemFont',
        'Segoe UI',
        'Roboto',
        'Helvetica Neue',
        'Arial',
        'sans-serif',
      ],
      // Technical identifiers only: addresses, ports, ids, scores, times.
      mono: [
        'ui-monospace',
        'JetBrains Mono',
        'SFMono-Regular',
        'Menlo',
        'Consolas',
        'Liberation Mono',
        'monospace',
      ],
    },
    fontSize: {
      '2xs': ['11px', { lineHeight: '16px' }],
      xs: ['12px', { lineHeight: '18px' }],
      sm: ['13px', { lineHeight: '20px' }],
      base: ['14px', { lineHeight: '21px' }],
      md: ['16px', { lineHeight: '24px' }],
      lg: ['20px', { lineHeight: '28px' }],
      xl: ['24px', { lineHeight: '32px' }],
    },
    borderRadius: {
      none: '0',
      sm: '4px',
      DEFAULT: '6px',
      md: '8px',
      full: '9999px',
    },
    boxShadow: {
      none: 'none',
      // One shadow. Anything that needs separation uses a border instead.
      overlay: 'var(--shadow)',
    },
    extend: {
      spacing: {
        sidebar: '224px',
        'sidebar-collapsed': '56px',
        header: '52px',
      },
      transitionDuration: {
        DEFAULT: '120ms',
      },
      keyframes: {
        // The only entrance animation in the product: a new table row.
        'row-in': {
          from: { opacity: '0', transform: 'translateY(-2px)' },
          to: { opacity: '1', transform: 'none' },
        },
        'overlay-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'panel-in': {
          from: { transform: 'translateX(8px)', opacity: '0' },
          to: { transform: 'none', opacity: '1' },
        },
      },
      animation: {
        'row-in': 'row-in 120ms ease-out',
        'overlay-in': 'overlay-in 120ms ease-out',
        'panel-in': 'panel-in 140ms ease-out',
      },
    },
  },
  plugins: [],
} satisfies Config
