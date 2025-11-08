/**
 * Главный компонент приложения
 * Настраивает роутинг и основную структуру приложения
 */
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import { useAppStore } from './store/useAppStore';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import Analysis from './pages/Analysis';
import Settings from './pages/Settings';

function App() {
  const theme = useAppStore((state) => state.theme);

  /**
   * Применяет тему при загрузке приложения
   */
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  return (
    <BrowserRouter>
      <div className="flex min-h-screen">
        {/* Боковое меню */}
        <Sidebar />

        {/* Основной контент */}
        <main className="flex-1 ml-64 min-h-screen">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
