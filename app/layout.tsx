import type { Metadata } from 'next';
import { Plus_Jakarta_Sans, JetBrains_Mono } from 'next/font/google';
import './globals.css';

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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${plusJakarta.variable} ${jetbrainsMono.variable}`}>
      <body
        suppressHydrationWarning
        className="bg-[#FAFAF8] text-[#202124] antialiased selection:bg-[#3478F6]/15 selection:text-[#3478F6]"
        style={{ fontFamily: 'var(--font-sans), sans-serif' }}
      >
        {children}
      </body>
    </html>
  );
}
