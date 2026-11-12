'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Eraser, Undo2, Sparkles, Grid, Eye } from 'lucide-react';
import { normalizeGridToMNIST28x28, synthesizeHandwrittenDigit } from '@/lib/neural-engine';
import { useTheme } from './ThemeProvider';

interface Point {
  x: number;
  y: number;
}

interface DrawingPadProps {
  pixels28x28: number[];
  onPixelsChange: (pixels: number[], autoRecognize?: boolean) => void;
  onRecognize: (pixels: number[]) => void;
}

const CANVAS_SIZE = 280;

export default function DrawingPad({
  pixels28x28,
  onPixelsChange,
  onRecognize,
}: DrawingPadProps) {
  const { isDark } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [strokes, setStrokes] = useState<Point[][]>([]);
  const currentStrokeRef = useRef<Point[]>([]);
  const isDrawingRef = useRef(false);
  const [showNumbersInGrid, setShowNumbersInGrid] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ r: number; c: number; val: number } | null>(
    null
  );
  const [hasDrawnFreehand, setHasDrawnFreehand] = useState(false);

  // Render strokes onto the 280x280 drawing canvas (adapting ink & guide colors to Light/Dark theme)
  const redrawCanvas = useCallback(
    (allStrokes: Point[][], activeStroke: Point[] = []) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

      // Subtle guide crosshair & safe drawing zone
      ctx.strokeStyle = isDark ? '#203653' : '#ECECE8';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(36, 36, CANVAS_SIZE - 72, CANVAS_SIZE - 72);
      ctx.beginPath();
      ctx.moveTo(CANVAS_SIZE / 2, 24);
      ctx.lineTo(CANVAS_SIZE / 2, CANVAS_SIZE - 24);
      ctx.moveTo(24, CANVAS_SIZE / 2);
      ctx.lineTo(CANVAS_SIZE - 24, CANVAS_SIZE / 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw smooth strokes
      const toDraw = [...allStrokes, ...(activeStroke.length > 0 ? [activeStroke] : [])];
      ctx.save();
      ctx.strokeStyle = isDark ? '#F1F6FF' : '#202124';
      ctx.fillStyle = isDark ? '#F1F6FF' : '#202124';
      if (isDark) {
        ctx.shadowColor = 'rgba(53, 216, 242, 0.45)';
        ctx.shadowBlur = 8;
      }
      ctx.lineWidth = 22;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const stroke of toDraw) {
        if (stroke.length === 0) continue;
        if (stroke.length === 1) {
          ctx.beginPath();
          ctx.arc(stroke[0].x, stroke[0].y, 11, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(stroke[0].x, stroke[0].y);
        for (let i = 1; i < stroke.length; i++) {
          const midX = (stroke[i - 1].x + stroke[i].x) / 2;
          const midY = (stroke[i - 1].y + stroke[i].y) / 2;
          ctx.quadraticCurveTo(stroke[i - 1].x, stroke[i - 1].y, midX, midY);
        }
        const last = stroke[stroke.length - 1];
        ctx.lineTo(last.x, last.y);
        ctx.stroke();
      }
      ctx.restore();

      // Paint 28x28 preview onto main canvas when a sample digit is active and no freehand strokes exist
      if (
        allStrokes.length === 0 &&
        activeStroke.length === 0 &&
        !hasDrawnFreehand &&
        pixels28x28.some((v) => v > 0.05)
      ) {
        const cell = CANVAS_SIZE / 28;
        for (let r = 0; r < 28; r++) {
          for (let c = 0; c < 28; c++) {
            const v = pixels28x28[r * 28 + c];
            if (v > 0.04) {
              ctx.fillStyle = isDark
                ? v > 0.6
                  ? `rgba(241, 246, 255, ${Math.min(1, v * 0.96)})`
                  : `rgba(53, 216, 242, ${Math.min(0.9, v * 0.85)})`
                : `rgba(32, 33, 36, ${Math.min(1, v * 0.95)})`;
              ctx.beginPath();
              ctx.arc(
                c * cell + cell / 2,
                r * cell + cell / 2,
                (cell / 2) * 1.15,
                0,
                Math.PI * 2
              );
              ctx.fill();
            }
          }
        }
      }
    },
    [isDark, hasDrawnFreehand, pixels28x28]
  );

  // Extract 28x28 normalized grayscale pixels from current strokes (theme-independent math)
  const extract28x28FromStrokes = useCallback((allStrokes: Point[][]): number[] => {
    if (allStrokes.length === 0) {
      return new Array(784).fill(0);
    }
    const offscreen = document.createElement('canvas');
    offscreen.width = 56;
    offscreen.height = 56;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return new Array(784).fill(0);

    let minX = CANVAS_SIZE;
    let maxX = 0;
    let minY = CANVAS_SIZE;
    let maxY = 0;
    for (const stroke of allStrokes) {
      for (const pt of stroke) {
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
      }
    }
    const span = Math.max(48, maxX - minX, maxY - minY);
    const adaptiveLineWidth = Math.max(6, span * 0.112);

    const scale = 56 / CANVAS_SIZE;
    ctx.strokeStyle = '#000000';
    ctx.fillStyle = '#000000';
    ctx.lineWidth = adaptiveLineWidth * scale;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const stroke of allStrokes) {
      if (stroke.length === 0) continue;
      if (stroke.length === 1) {
        ctx.beginPath();
        ctx.arc(
          stroke[0].x * scale,
          stroke[0].y * scale,
          (adaptiveLineWidth / 2) * scale,
          0,
          Math.PI * 2
        );
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(stroke[0].x * scale, stroke[0].y * scale);
      for (let i = 1; i < stroke.length; i++) {
        const midX = ((stroke[i - 1].x + stroke[i].x) / 2) * scale;
        const midY = ((stroke[i - 1].y + stroke[i].y) / 2) * scale;
        ctx.quadraticCurveTo(
          stroke[i - 1].x * scale,
          stroke[i - 1].y * scale,
          midX,
          midY
        );
      }
      const last = stroke[stroke.length - 1];
      ctx.lineTo(last.x * scale, last.y * scale);
      ctx.stroke();
    }

    const imgData = ctx.getImageData(0, 0, 56, 56).data;
    const rawAlpha = new Float32Array(56 * 56);
    for (let i = 0; i < 56 * 56; i++) {
      rawAlpha[i] = imgData[i * 4 + 3] / 255.0;
    }

    return normalizeGridToMNIST28x28(rawAlpha, 56, 56);
  }, []);

  useEffect(() => {
    redrawCanvas(strokes);
  }, [strokes, redrawCanvas]);

  const getPointerPos = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    isDrawingRef.current = true;
    setHasDrawnFreehand(true);
    const pt = getPointerPos(e);
    currentStrokeRef.current = [pt];
    redrawCanvas(strokes, currentStrokeRef.current);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    e.preventDefault();
    const pt = getPointerPos(e);
    currentStrokeRef.current.push(pt);
    redrawCanvas(strokes, currentStrokeRef.current);
  };

  const finishStroke = () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    if (currentStrokeRef.current.length > 0) {
      const updated = [...strokes, [...currentStrokeRef.current]];
      currentStrokeRef.current = [];
      setStrokes(updated);
      const extracted = extract28x28FromStrokes(updated);
      onPixelsChange(extracted, true);
    }
  };

  const handleUndo = () => {
    if (strokes.length === 0) return;
    const updated = strokes.slice(0, -1);
    setStrokes(updated);
    if (updated.length === 0) {
      setHasDrawnFreehand(false);
      onPixelsChange(new Array(784).fill(0), false);
    } else {
      const extracted = extract28x28FromStrokes(updated);
      onPixelsChange(extracted, true);
    }
  };

  const handleClear = () => {
    setStrokes([]);
    currentStrokeRef.current = [];
    setHasDrawnFreehand(false);
    redrawCanvas([]);
    onPixelsChange(new Array(784).fill(0), false);
  };

  const sampleVariationRef = useRef(0);

  const handleLoadSampleDigit = (digit: number) => {
    setStrokes([]);
    setHasDrawnFreehand(false);
    sampleVariationRef.current += 1;
    const v = sampleVariationRef.current;
    const sample = synthesizeHandwrittenDigit(digit, {
      thickness: 1.25,
      slant: ((v % 5) - 2) * 0.04,
      variant: v % 8,
    });
    onPixelsChange(sample, true);
  };

  const activePixelsCount = pixels28x28.filter((p) => p > 0.1).length;

  return (
    <div className="bg-white rounded-2xl border border-[#E5E5E0] p-6 transition-all">
      <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
        <div>
          <p className="text-xs font-medium text-[#777777] tracking-wide mb-1">
            Stage 1 · Input Canvas
          </p>
          <h2 className="text-xl font-semibold text-[#202124]">
            Draw any number from 0 to 9
          </h2>
          <p className="text-sm text-[#777777] mt-0.5">
            Use your finger or mouse inside the square, or tap a sample number below.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleUndo}
            disabled={strokes.length === 0}
            aria-label="Undo last stroke"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border border-[#E5E5E0] text-[#202124] bg-[#FAFAF8] hover:bg-[#F2F2EF] disabled:opacity-40 disabled:pointer-events-none transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Undo2 className="w-3.5 h-3.5" />
            Undo
          </button>
          <button
            type="button"
            onClick={handleClear}
            aria-label="Clear drawing canvas"
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium border border-[#E5E5E0] text-[#202124] bg-[#FAFAF8] hover:bg-[#F2F2EF] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Eraser className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>
      </div>

      {/* Main Side-by-Side: Drawing Canvas + 28x28 Pixel Matrix */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 items-stretch">
        {/* Interactive Canvas Column */}
        <div className="flex flex-col items-center justify-between bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4">
          <div
            className={`relative w-full max-w-[240px] aspect-square rounded-xl border overflow-hidden shadow-2xs touch-none ${
              isDark
                ? 'bg-[#09172B] border-[#29476A] shadow-[inset_0_0_24px_rgba(5,11,24,0.65)]'
                : 'bg-white border-[#E5E5E0]'
            }`}
          >
            <canvas
              ref={canvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              data-testid="digit-drawing-canvas"
              aria-label="Interactive drawing canvas for digits 0 to 9"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={finishStroke}
              onPointerCancel={finishStroke}
              onPointerLeave={finishStroke}
              className="w-full h-full cursor-crosshair touch-none"
            />
            {activePixelsCount === 0 && (
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center p-4">
                <span
                  className={`text-4xl font-semibold select-none ${
                    isDark ? 'text-[#7186A6]/35' : 'text-[#202124]/15'
                  }`}
                >
                  0 – 9
                </span>
                <span className="text-xs text-[#777777] mt-1">
                  Draw a large single digit here
                </span>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={() => onRecognize(pixels28x28)}
            disabled={activePixelsCount === 0}
            className="mt-4 w-full py-2.5 px-4 rounded-xl bg-[#3478F6] hover:bg-[#2563EB] disabled:bg-[#E5E5E0] disabled:text-[#777777] text-white font-medium text-sm flex items-center justify-center gap-2 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            Recognise Number
          </button>
        </div>

        {/* 28 x 28 Numerical Pixel Representation Column */}
        <div className="flex flex-col justify-between bg-[#FAFAF8] rounded-xl border border-[#E5E5E0] p-4">
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="min-w-0">
              <span className="text-xs font-semibold text-[#202124] block whitespace-nowrap">
                28 × 28 Grid
              </span>
              <p className="text-[11px] text-[#777777] tabular-nums">
                784 numbers · {activePixelsCount} ink cells
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowNumbersInGrid((v) => !v)}
              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                showNumbersInGrid
                  ? 'bg-[#3478F6] text-white border-[#3478F6]'
                  : 'bg-white text-[#202124] border-[#E5E5E0] hover:bg-[#F2F2EF]'
              }`}
            >
              {showNumbersInGrid ? (
                <>
                  <Eye className="w-3 h-3" />
                  Pixel View
                </>
              ) : (
                <>
                  <Grid className="w-3 h-3" />
                  Number View
                </>
              )}
            </button>
          </div>

          {/* 28x28 Interactive Grid */}
          <div
            className={`w-full max-w-[216px] mx-auto aspect-square rounded-lg border p-1 grid ${
              isDark ? 'bg-[#09172B] border-[#29476A]' : 'bg-white border-[#E5E5E0]'
            }`}
            style={{
              gridTemplateColumns: 'repeat(28, minmax(0, 1fr))',
              gridTemplateRows: 'repeat(28, minmax(0, 1fr))',
            }}
            onMouseLeave={() => setHoveredCell(null)}
          >
            {pixels28x28.map((val, idx) => {
              const r = Math.floor(idx / 28);
              const c = idx % 28;
              const active = val > 0.04;
              const cellBg = active
                ? isDark
                  ? val > 0.65
                    ? `rgba(53, 216, 242, ${Math.min(1, val)})`
                    : `rgba(37, 139, 255, ${Math.max(0.18, val)})`
                  : `rgba(52, 120, 246, ${Math.max(0.12, val)})`
                : isDark
                ? idx % 2 === 0
                  ? '#09172B'
                  : '#0B1B32'
                : idx % 2 === 0
                ? '#FFFFFF'
                : '#FAFAF8';

              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredCell({ r, c, val })}
                  className="relative flex items-center justify-center select-none overflow-hidden"
                  style={{ backgroundColor: cellBg }}
                >
                  {showNumbersInGrid && val >= 0.15 && (
                    <span className="text-[5px] leading-none font-mono text-white font-semibold tabular-nums">
                      {Math.round(val * 9)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          <div className="mt-3 pt-2.5 border-t border-[#E5E5E0] flex items-center justify-between text-[11px] text-[#777777]">
            {hoveredCell ? (
              <span className="font-mono text-[#202124] tabular-nums">
                Row {hoveredCell.r + 1}, Col {hoveredCell.c + 1} · Value:{' '}
                <strong>{hoveredCell.val.toFixed(2)}</strong>
              </span>
            ) : (
              <span>
                Your drawing is converted into numbers (0.00 to 1.00) the AI can read.
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Quick Sample Digits Bar for Touch / Exhibition Convenience */}
      <div className="mt-4 pt-4 border-t border-[#E5E5E0] flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs font-medium text-[#777777]">
          Or test a sample handwritten digit:
        </span>
        <div className="flex flex-wrap items-center gap-1.5">
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => handleLoadSampleDigit(d)}
              aria-label={`Load sample digit ${d}`}
              className="w-8 h-8 rounded-lg border border-[#E5E5E0] bg-[#FAFAF8] hover:bg-[#3478F6] hover:text-white hover:border-[#3478F6] text-xs font-semibold text-[#202124] transition-colors tabular-nums cursor-pointer"
            >
              {d}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
