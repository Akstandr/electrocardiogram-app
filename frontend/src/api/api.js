/**
 * API модуль для взаимодействия с сервером
 * Содержит функции для отправки файлов и получения результатов анализа ЭКГ
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

/**
 * Отправляет JSON-файл с данными ЭКГ на сервер для анализа
 * @param {File} file - JSON файл с данными ЭКГ
 * @returns {Promise<{id: string, status: string}>} - ID задачи и статус обработки
 */
export async function sendFileToServer(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error sending file to server:', error);
    throw error;
  }
}

/**
 * Получает результат анализа ЭКГ по ID задачи
 * @param {string} id - ID задачи анализа
 * @returns {Promise<{result: string, status: string}>} - Результат анализа и статус
 */
export async function getAnalysisResult(id) {
  try {
    const response = await fetch(`${API_BASE_URL}/analysis/${id}`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting analysis result:', error);
    throw error;
  }
}

