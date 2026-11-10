export type ThemeMode = 'light' | 'dark';

export const THEME_STORAGE_KEY = 'neurolab_theme_preference';

export interface ThemePalette {
  mode: ThemeMode;
  bgPrimary: string;
  bgSecondary: string;
  bgCard: string;
  bgElevated: string;
  bgInput: string;
  accentBlue: string;
  accentCyan: string;
  accentViolet: string;
  accentSoftBlue: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  borderSubtle: string;
  borderElevated: string;
}

export const DARK_MODE_PALETTE: ThemePalette = {
  mode: 'dark',
  bgPrimary: '#050B18',
  bgSecondary: '#081326',
  bgCard: '#0D1B33',
  bgElevated: '#122440',
  bgInput: '#09172B',
  accentBlue: '#258BFF',
  accentCyan: '#35D8F2',
  accentViolet: '#745BFF',
  accentSoftBlue: '#579BFF',
  textPrimary: '#F1F6FF',
  textSecondary: '#A2B5D0',
  textMuted: '#7186A6',
  borderSubtle: '#203653',
  borderElevated: '#29476A',
};

export const LIGHT_MODE_PALETTE: ThemePalette = {
  mode: 'light',
  bgPrimary: '#FAFAF8',
  bgSecondary: '#F2F2EF',
  bgCard: '#FFFFFF',
  bgElevated: '#FAFAF8',
  bgInput: '#FFFFFF',
  accentBlue: '#3478F6',
  accentCyan: '#3478F6',
  accentViolet: '#5856D6',
  accentSoftBlue: '#3478F6',
  textPrimary: '#202124',
  textSecondary: '#777777',
  textMuted: '#777777',
  borderSubtle: '#E5E5E0',
  borderElevated: '#D4D4CE',
};

export function resolveInitialTheme(
  storedPreference: string | null | undefined,
  systemPrefersDark: boolean
): ThemeMode {
  if (storedPreference === 'dark' || storedPreference === 'light') {
    return storedPreference;
  }
  return systemPrefersDark ? 'dark' : 'light';
}

export function getNextTheme(current: ThemeMode): ThemeMode {
  return current === 'dark' ? 'light' : 'dark';
}

export function getThemePalette(mode: ThemeMode): ThemePalette {
  return mode === 'dark' ? DARK_MODE_PALETTE : LIGHT_MODE_PALETTE;
}
