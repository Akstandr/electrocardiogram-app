/**
 * Страница настроек
 * Позволяет выбрать тему (light/dark) и язык интерфейса (RU/EN/DE)
 */
import { useAppStore } from '../store/useAppStore';

const translations = {
  ru: {
    title: 'Настройки',
    theme: 'Тема',
    themeLight: 'Светлая',
    themeDark: 'Тёмная',
    language: 'Язык интерфейса',
    languageRu: 'Русский',
    languageEn: 'English',
    languageDe: 'Deutsch',
    description: 'Выберите предпочтительную тему и язык интерфейса.',
  },
  en: {
    title: 'Settings',
    theme: 'Theme',
    themeLight: 'Light',
    themeDark: 'Dark',
    language: 'Interface Language',
    languageRu: 'Русский',
    languageEn: 'English',
    languageDe: 'Deutsch',
    description: 'Choose your preferred theme and interface language.',
  },
  de: {
    title: 'Einstellungen',
    theme: 'Design',
    themeLight: 'Hell',
    themeDark: 'Dunkel',
    language: 'Sprache der Benutzeroberfläche',
    languageRu: 'Русский',
    languageEn: 'English',
    languageDe: 'Deutsch',
    description: 'Wählen Sie Ihr bevorzugtes Design und die Sprache der Benutzeroberfläche.',
  },
};

export default function Settings() {
  const language = useAppStore((state) => state.language);
  const theme = useAppStore((state) => state.theme);
  const setTheme = useAppStore((state) => state.setTheme);
  const setLanguage = useAppStore((state) => state.setLanguage);

  const t = translations[language];

  /**
   * Применяет выбранную тему к документу
   */
  const applyTheme = (newTheme) => {
    setTheme(newTheme);
    if (newTheme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <main
      className={`min-h-screen transition-colors duration-200 ${
        theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'
      }`}
    >
      <div className="max-w-2xl mx-auto p-8">
        <h1
          className={`text-3xl font-bold mb-6 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}
        >
          {t.title}
        </h1>

        <p
          className={`mb-8 text-lg ${
            theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
          }`}
        >
          {t.description}
        </p>

        <div className="space-y-8">
          {/* Выбор темы */}
          <section
            className={`p-6 rounded-2xl border shadow-lg ${
              theme === 'dark'
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}
          >
            <h2
              className={`text-xl font-semibold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.theme}
            </h2>
            <div className="flex gap-4">
              <button
                onClick={() => applyTheme('light')}
                className={`flex-1 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  theme === 'light'
                    ? 'bg-blue-600 text-white shadow-lg hover:bg-blue-700'
                    : theme === 'dark'
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
                }`}
              >
                {t.themeLight}
              </button>
              <button
                onClick={() => applyTheme('dark')}
                className={`flex-1 px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  theme === 'dark'
                    ? 'bg-blue-600 text-white shadow-lg hover:bg-blue-700'
                    : theme === 'dark'
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
                }`}
              >
                {t.themeDark}
              </button>
            </div>
          </section>

          {/* Выбор языка */}
          <section
            className={`p-6 rounded-2xl border shadow-lg ${
              theme === 'dark'
                ? 'bg-gray-800 border-gray-700'
                : 'bg-white border-gray-200'
            }`}
          >
            <h2
              className={`text-xl font-semibold mb-4 ${
                theme === 'dark' ? 'text-white' : 'text-gray-900'
              }`}
            >
              {t.language}
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <button
                onClick={() => setLanguage('ru')}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  language === 'ru'
                    ? 'bg-blue-600 text-white shadow-lg hover:bg-blue-700'
                    : theme === 'dark'
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
                }`}
              >
                {t.languageRu}
              </button>
              <button
                onClick={() => setLanguage('en')}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  language === 'en'
                    ? 'bg-blue-600 text-white shadow-lg hover:bg-blue-700'
                    : theme === 'dark'
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
                }`}
              >
                {t.languageEn}
              </button>
              <button
                onClick={() => setLanguage('de')}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 ${
                  language === 'de'
                    ? 'bg-blue-600 text-white shadow-lg hover:bg-blue-700'
                    : theme === 'dark'
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
                }`}
              >
                {t.languageDe}
              </button>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

