/**
 * Боковое меню навигации
 * Фиксированное меню слева с пунктами для перехода между разделами
 */
import { Link, useLocation } from 'react-router-dom';
import { useAppStore } from '../store/useAppStore';

const translations = {
  ru: {
    home: 'Главная',
    analysis: 'Анализ',
    settings: 'Настройки',
  },
  en: {
    home: 'Home',
    analysis: 'Analysis',
    settings: 'Settings',
  },
  de: {
    home: 'Startseite',
    analysis: 'Analyse',
    settings: 'Einstellungen',
  },
};

export default function Sidebar() {
  const location = useLocation();
  const language = useAppStore((state) => state.language);
  const theme = useAppStore((state) => state.theme);

  const t = translations[language];

  const navItems = [
    { path: '/', label: t.home, icon: '🏠' },
    { path: '/analysis', label: t.analysis, icon: '📊' },
    { path: '/settings', label: t.settings, icon: '⚙️' },
  ];

  return (
    <aside
      className={`fixed left-0 top-0 h-screen w-64 z-50 border-r shadow-lg transition-colors duration-200 ${
        theme === 'dark'
          ? 'bg-gray-900 border-gray-700'
          : 'bg-white border-gray-200'
      }`}
    >
      <div className="p-6 h-full overflow-y-auto">
        <h1
          className={`text-2xl font-bold mb-8 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}
        >
          ECG Analyzer
        </h1>
        <nav className="space-y-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 shadow-sm ${
                  isActive
                    ? theme === 'dark'
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'bg-blue-100 text-blue-700 shadow-md'
                    : theme === 'dark'
                    ? 'text-gray-300 hover:bg-gray-800 hover:text-white hover:shadow-md'
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900 hover:shadow-md'
                }`}
              >
                <span className="text-xl">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}

