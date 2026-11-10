import type { Metadata } from 'next';
import { Plus_Jakarta_Sans, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/ThemeProvider';

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'NEUROLAB — Interactive Machine Learning Experience',
  description:
    'Teach a machine. Watch it learn. An interactive machine learning exhibition experience for drawing digits, inspecting a real 784-128-64-10 neural network, and teaching AI through live backpropagation.',
  openGraph: {
    title: 'NEUROLAB — Interactive Machine Learning Experience',
    description:
      'Teach a machine. Watch it learn. An interactive machine learning exhibition experience for drawing digits, inspecting a real 784-128-64-10 neural network, and teaching AI through live backpropagation.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NEUROLAB — Interactive Machine Learning Experience',
    description:
      'Teach a machine. Watch it learn. An interactive machine learning exhibition experience for drawing digits, inspecting a real 784-128-64-10 neural network, and teaching AI through live backpropagation.',
  },
};

const themeInitScript = `
(function() {
  try {
    var stored = localStorage.getItem('neurolab_theme_preference');
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var mode = (stored === 'dark' || stored === 'light') ? stored : (sysDark ? 'dark' : 'light');
    var root = document.documentElement;
    if (mode === 'dark') {
      root.classList.add('dark');
      root.setAttribute('data-theme', 'dark');
      root.style.colorScheme = 'dark';
    } else {
      root.classList.remove('dark');
      root.setAttribute('data-theme', 'light');
      root.style.colorScheme = 'light';
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${plusJakarta.variable} ${jetbrainsMono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body
        suppressHydrationWarning
        className="bg-[#FAFAF8] dark:bg-[#050B18] text-[#202124] dark:text-[#F1F6FF] antialiased selection:bg-[#3478F6]/15 selection:text-[#3478F6]"
        style={{ fontFamily: 'var(--font-sans), sans-serif' }}
      >
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
