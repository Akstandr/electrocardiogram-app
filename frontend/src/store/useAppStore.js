/**
 * Zustand store для управления глобальным состоянием приложения
 * Управляет темой (light/dark) и языком интерфейса (RU/EN/DE)
 */
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export const useAppStore = create(
  persist(
    (set) => ({
      // Тема приложения: 'light' или 'dark'
      theme: 'light',
      
      // Язык интерфейса: 'ru', 'en' или 'de'
      language: 'ru',

      // Установка темы
      setTheme: (theme) => set({ theme }),

      // Установка языка
      setLanguage: (language) => set({ language }),
    }),
    {
      name: 'ecg-app-storage', // Имя для localStorage
      storage: createJSONStorage(() => localStorage),
    }
  )
);

