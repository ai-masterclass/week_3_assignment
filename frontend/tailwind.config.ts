import type { Config } from "tailwindcss"
import { brandColors, neutralColors, semanticColors } from "./src/config/design"

const config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    screens: {
      'xs': '400px',
      'sm': '640px',
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
    },
    extend: {
      fontFamily: {
        heading: ['var(--font-mondwest-bold)', 'sans-serif'],
        body: ['var(--font-mondwest-regular)', 'monospace'],
        sans: ['var(--font-mondwest-regular)', 'monospace'],
      },
      colors: {
        brand: {
          primary: brandColors.primary,
          secondary: brandColors.secondary,
          yellow: brandColors.yellow,
          orange: brandColors.orange,
          blue: brandColors.blue,
        },
        slate: neutralColors.slate,
        success: semanticColors.success,
        error: semanticColors.error,
        warning: semanticColors.warning,
        info: semanticColors.info,
      },
      backgroundImage: {
        'gradient-primary': 'radial-gradient(circle at 0% 0%, #FDB44B 0%, #FB923C 15%, #FF6B35 28%, #F97316 38%, #3B82F6 55%, #2563EB 70%, #1E40AF 85%)',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-up': 'slide-up 0.3s ease-out',
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config

export default config

