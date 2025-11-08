/**
 * Компонент для отображения изображения ЭКГ с возможностью зуммирования
 * Использует библиотеку react-zoom-pan-pinch для интерактивного просмотра
 */
import { useState } from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { useAppStore } from '../store/useAppStore';

export default function ECGViewer({ imageUrl }) {
  const theme = useAppStore((state) => state.theme);
  const language = useAppStore((state) => state.language);
  const [zoomLevel, setZoomLevel] = useState(100); // Начальный уровень зума в процентах

  // Если изображение не передано, используем заглушку
  const defaultImageUrl =
    'https://via.placeholder.com/800x400/4F46E5/FFFFFF?text=ECG+Image';

  const tooltips = {
    ru: {
      zoomIn: 'Увеличить',
      zoomOut: 'Уменьшить',
      reset: 'Сбросить',
    },
    en: {
      zoomIn: 'Zoom in',
      zoomOut: 'Zoom out',
      reset: 'Reset',
    },
    de: {
      zoomIn: 'Vergrößern',
      zoomOut: 'Verkleinern',
      reset: 'Zurücksetzen',
    },
  };

  const t = tooltips[language] || tooltips.en;

  return (
    <div
      className={`w-full h-full rounded-2xl border overflow-hidden shadow-lg ${
        theme === 'dark'
          ? 'bg-gray-800 border-gray-700'
          : 'bg-white border-gray-200'
      }`}
    >
      <TransformWrapper
        initialScale={1}
        minScale={0.5}
        maxScale={3}
        wheel={{ step: 0.1 }}
        pan={{ disabled: false }}
        doubleClick={{ disabled: false }}
        onTransformed={(ref, state) => {
          // Обновляем уровень зума при изменении масштаба
          const scale = state.scale || 1;
          setZoomLevel(Math.round(scale * 100));
        }}
      >
        {({ zoomIn, zoomOut, resetTransform }) => {
          return (
            <>
              {/* Панель управления зумом */}
              <div
                className={`absolute top-4 right-4 z-10 flex items-center gap-2 ${
                  theme === 'dark' ? 'bg-gray-800' : 'bg-white'
                } rounded-xl shadow-xl p-2 border ${
                  theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
                }`}
              >
                <button
                  onClick={() => zoomIn()}
                  className={`px-3 py-1 rounded-lg transition-all duration-200 ${
                    theme === 'dark'
                      ? 'bg-gray-700 hover:bg-gray-600 text-white hover:shadow-md'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700 hover:shadow-md'
                  }`}
                  title={t.zoomIn}
                >
                  +
                </button>
                <span
                  className={`px-2 py-1 text-sm font-semibold min-w-[3rem] text-center ${
                    theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                  }`}
                >
                  {zoomLevel}%
                </span>
                <button
                  onClick={() => zoomOut()}
                  className={`px-3 py-1 rounded-lg transition-all duration-200 ${
                    theme === 'dark'
                      ? 'bg-gray-700 hover:bg-gray-600 text-white hover:shadow-md'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700 hover:shadow-md'
                  }`}
                  title={t.zoomOut}
                >
                  −
                </button>
                <button
                  onClick={() => {
                    resetTransform();
                    setZoomLevel(100);
                  }}
                  className={`px-3 py-1 rounded-lg transition-all duration-200 ${
                    theme === 'dark'
                      ? 'bg-gray-700 hover:bg-gray-600 text-white hover:shadow-md'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700 hover:shadow-md'
                  }`}
                  title={t.reset}
                >
                  ↺
                </button>
              </div>

              {/* Область с изображением */}
              <TransformComponent
                wrapperClass="w-full h-full"
                contentClass="w-full h-full flex items-center justify-center"
              >
                <img
                  src={imageUrl || defaultImageUrl}
                  alt="ECG"
                  className="max-w-full max-h-full object-contain"
                />
              </TransformComponent>
            </>
          );
        }}
      </TransformWrapper>
    </div>
  );
}

