/**
 * Главная страница приложения
 * Содержит краткое описание и документацию приложения
 */
import { useAppStore } from '../store/useAppStore';

const translations = {
  ru: {
    title: 'Добро пожаловать в ECG Analyzer',
    subtitle: 'Система анализа электрокардиограмм с помощью искусственного интеллекта',
    description: {
      heading: 'О приложении',
      text: 'ECG Analyzer — это современное веб-приложение для анализа электрокардиограмм (ЭКГ) с использованием технологий искусственного интеллекта. Приложение позволяет загружать данные ЭКГ в формате JSON и получать детальный анализ с визуализацией результатов.',
    },
    features: {
      heading: 'Основные возможности',
      items: [
        'Загрузка и обработка данных ЭКГ в формате JSON',
        'Автоматический анализ с помощью ИИ',
        'Интерактивная визуализация ЭКГ с возможностью зуммирования',
        'Детальное заключение по результатам анализа',
        'Поддержка светлой и тёмной темы',
        'Многоязычный интерфейс (русский/английский/немецкий)',
      ],
    },
    usage: {
      heading: 'Как использовать',
      text: 'Перейдите в раздел "Анализ" для загрузки файла с данными ЭКГ. После загрузки файла система автоматически обработает данные и предоставит результаты анализа.',
    },
  },
  en: {
    title: 'Welcome to ECG Analyzer',
    subtitle: 'Electrocardiogram analysis system powered by artificial intelligence',
    description: {
      heading: 'About the application',
      text: 'ECG Analyzer is a modern web application for analyzing electrocardiograms (ECG) using artificial intelligence technologies. The application allows you to upload ECG data in JSON format and receive detailed analysis with visualization of results.',
    },
    features: {
      heading: 'Key features',
      items: [
        'Upload and process ECG data in JSON format',
        'Automatic AI-powered analysis',
        'Interactive ECG visualization with zoom capability',
        'Detailed analysis report',
        'Light and dark theme support',
        'Multilingual interface (Russian/English/German)',
      ],
    },
    usage: {
      heading: 'How to use',
      text: 'Go to the "Analysis" section to upload an ECG data file. After uploading the file, the system will automatically process the data and provide analysis results.',
    },
  },
  de: {
    title: 'Willkommen bei ECG Analyzer',
    subtitle: 'Elektrokardiogramm-Analysesystem mit künstlicher Intelligenz',
    description: {
      heading: 'Über die Anwendung',
      text: 'ECG Analyzer ist eine moderne Webanwendung zur Analyse von Elektrokardiogrammen (EKG) mit Technologien der künstlichen Intelligenz. Die Anwendung ermöglicht es Ihnen, EKG-Daten im JSON-Format hochzuladen und eine detaillierte Analyse mit Visualisierung der Ergebnisse zu erhalten.',
    },
    features: {
      heading: 'Hauptfunktionen',
      items: [
        'Hochladen und Verarbeiten von EKG-Daten im JSON-Format',
        'Automatische KI-gestützte Analyse',
        'Interaktive EKG-Visualisierung mit Zoom-Funktion',
        'Detaillierter Analysebericht',
        'Unterstützung für helle und dunkle Themen',
        'Mehrsprachige Benutzeroberfläche (Russisch/Englisch/Deutsch)',
      ],
    },
    usage: {
      heading: 'Wie man verwendet',
      text: 'Gehen Sie zum Abschnitt "Analyse", um eine EKG-Datendatei hochzuladen. Nach dem Hochladen der Datei verarbeitet das System die Daten automatisch und stellt die Analyseergebnisse bereit.',
    },
  },
};

export default function Home() {
  const language = useAppStore((state) => state.language);
  const theme = useAppStore((state) => state.theme);
  const t = translations[language];

  return (
    <main
      className={`min-h-screen transition-colors duration-200 ${
        theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'
      }`}
    >
      <div className="max-w-4xl mx-auto p-8 pt-12">
        <section className="mb-12">
          <div
            className={`rounded-2xl p-8 shadow-lg mb-6 ${
              theme === 'dark'
                ? 'bg-gray-800 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <h1
              className={`text-4xl font-bold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.title}
            </h1>
            <p
              className={`text-xl ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
              }`}
            >
              {t.subtitle}
            </p>
          </div>
        </section>

        <section className="mb-12">
          <div
            className={`rounded-2xl p-8 shadow-lg ${
              theme === 'dark'
                ? 'bg-gray-800 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <h2
              className={`text-2xl font-semibold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.description.heading}
            </h2>
            <p
              className={`text-lg leading-relaxed ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
              }`}
            >
              {t.description.text}
            </p>
          </div>
        </section>

        <section className="mb-12">
          <div
            className={`rounded-2xl p-8 shadow-lg ${
              theme === 'dark'
                ? 'bg-gray-800 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <h2
              className={`text-2xl font-semibold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.features.heading}
            </h2>
            <ul
              className={`space-y-3 ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
              }`}
            >
              {t.features.items.map((item, index) => (
                <li
                  key={index}
                  className={`text-lg flex items-start gap-3 ${
                    theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                  }`}
                >
                  <span className="text-blue-500 text-xl">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section>
          <div
            className={`rounded-2xl p-8 shadow-lg ${
              theme === 'dark'
                ? 'bg-gray-800 border border-gray-700'
                : 'bg-white border border-gray-200'
            }`}
          >
            <h2
              className={`text-2xl font-semibold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.usage.heading}
            </h2>
            <p
              className={`text-lg leading-relaxed ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
              }`}
            >
              {t.usage.text}
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}

