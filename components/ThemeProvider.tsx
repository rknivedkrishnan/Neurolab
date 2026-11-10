'use client';

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from 'react';
import {
  ThemeMode,
  ThemePalette,
  THEME_STORAGE_KEY,
  resolveInitialTheme,
  getNextTheme,
  getThemePalette,
} from '@/lib/theme-engine';

interface ThemeContextValue {
  theme: ThemeMode;
  isDark: boolean;
  palette: ThemePalette;
  toggleTheme: () => void;
  setTheme: (next: ThemeMode) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'light',
  isDark: false,
  palette: getThemePalette('light'),
  toggleTheme: () => {},
  setTheme: () => {},
});

function applyThemeToDOM(mode: ThemeMode) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (mode === 'dark') {
    root.classList.add('dark');
    root.setAttribute('data-theme', 'dark');
    root.style.colorScheme = 'dark';
  } else {
    root.classList.remove('dark');
    root.setAttribute('data-theme', 'light');
    root.style.colorScheme = 'light';
  }
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>('light');

  // Hydrate initial theme from DOM / localStorage / OS preference on mount
  useEffect(() => {
    queueMicrotask(() => {
      try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        const sysDark =
          typeof window !== 'undefined' &&
          window.matchMedia?.('(prefers-color-scheme: dark)').matches;
        const resolved = resolveInitialTheme(stored, Boolean(sysDark));
        setThemeState(resolved);
        applyThemeToDOM(resolved);
      } catch {
        setThemeState('light');
      }
    });
  }, []);

  // Listen for OS colour scheme changes if the user hasn't saved an explicit preference
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (stored !== 'dark' && stored !== 'light') {
          const next: ThemeMode = e.matches ? 'dark' : 'light';
          setThemeState(next);
          applyThemeToDOM(next);
        }
      } catch {
        // Ignore storage errors
      }
    };
    mql.addEventListener('change', handleChange);
    return () => mql.removeEventListener('change', handleChange);
  }, []);

  const setTheme = useCallback((next: ThemeMode) => {
    setThemeState(next);
    applyThemeToDOM(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Ignore storage errors
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => {
      const next = getNextTheme(prev);
      applyThemeToDOM(next);
      try {
        localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        // Ignore storage errors
      }
      return next;
    });
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme,
      isDark: theme === 'dark',
      palette: getThemePalette(theme),
      toggleTheme,
      setTheme,
    }),
    [theme, toggleTheme, setTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
