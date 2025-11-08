/**
 * Страница анализа ЭКГ
 * Позволяет загрузить JSON-файл с данными ЭКГ и отобразить результаты анализа
 */
import { useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { sendFileToServer, getAnalysisResult } from '../api/api';
import ECGViewer from '../components/ECGViewer';

const translations = {
  ru: {
    title: 'Анализ ЭКГ',
    uploadFile: 'Загрузить JSON-файл',
    currentFile: 'Текущий файл:',
    noFile: 'Файл не загружен',
    conclusion: 'Заключение',
    conclusionPlaceholder: 'Здесь будет результат анализа',
    loading: 'Обработка...',
    error: 'Ошибка при загрузке файла',
    ecgImage: 'Изображение ЭКГ',
  },
  en: {
    title: 'ECG Analysis',
    uploadFile: 'Upload JSON file',
    currentFile: 'Current file:',
    noFile: 'No file uploaded',
    conclusion: 'Conclusion',
    conclusionPlaceholder: 'Analysis results will appear here',
    loading: 'Processing...',
    error: 'Error uploading file',
    ecgImage: 'ECG Image',
  },
  de: {
    title: 'EKG-Analyse',
    uploadFile: 'JSON-Datei hochladen',
    currentFile: 'Aktuelle Datei:',
    noFile: 'Keine Datei hochgeladen',
    conclusion: 'Zusammenfassung',
    conclusionPlaceholder: 'Hier werden die Analyseergebnisse angezeigt',
    loading: 'Verarbeitung...',
    error: 'Fehler beim Hochladen der Datei',
    ecgImage: 'EKG-Bild',
  },
};

export default function Analysis() {
  const language = useAppStore((state) => state.language);
  const theme = useAppStore((state) => state.theme);
  const t = translations[language];

  const [selectedFile, setSelectedFile] = useState(null);
  const [conclusion, setConclusion] = useState(t.conclusionPlaceholder);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ecgImageUrl, setEcgImageUrl] = useState(null);

  /**
   * Обработчик загрузки файла
   */
  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Проверка типа файла
    if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
      const errorMessages = {
        ru: 'Пожалуйста, выберите JSON-файл',
        en: 'Please select a JSON file',
        de: 'Bitte wählen Sie eine JSON-Datei aus',
      };
      setError(errorMessages[language] || errorMessages.en);
      return;
    }

    setSelectedFile(file);
    setError(null);
    setIsLoading(true);
    setConclusion(t.loading);

    try {
      // Отправка файла на сервер
      const response = await sendFileToServer(file);
      
      // Если сервер вернул ID задачи, можно опросить результат
      if (response.id) {
        // В реальном приложении здесь можно использовать polling или WebSocket
        // Для демонстрации просто устанавливаем заглушку
        setTimeout(() => {
          const successMessages = {
            ru: `Анализ завершён. ID задачи: ${response.id}\n\nЗдесь будет детальное заключение по результатам анализа ЭКГ.`,
            en: `Analysis completed. Task ID: ${response.id}\n\nDetailed conclusion based on ECG analysis results will appear here.`,
            de: `Analyse abgeschlossen. Aufgaben-ID: ${response.id}\n\nHier wird eine detaillierte Zusammenfassung der EKG-Analyseergebnisse angezeigt.`,
          };
          setConclusion(successMessages[language] || successMessages.en);
          setIsLoading(false);
        }, 2000);
      } else {
        // Если сервер сразу вернул результат
        setConclusion(response.result || t.conclusionPlaceholder);
        setIsLoading(false);
      }

      // В реальном приложении здесь можно установить URL изображения ЭКГ
      // setEcgImageUrl(response.imageUrl);
    } catch (err) {
      console.error('Error processing file:', err);
      setError(err.message || t.error);
      setConclusion(t.conclusionPlaceholder);
      setIsLoading(false);
    }
  };

  return (
    <main
      className={`min-h-screen transition-colors duration-200 ${
        theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'
      }`}
    >
      <div className="p-8">
        <h1
          className={`text-3xl font-bold mb-6 ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}
        >
          {t.title}
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Левая колонка: загрузка файла и заключение */}
          <div className="space-y-6">
            {/* Загрузка файла */}
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
                {t.uploadFile}
              </h2>
              <div className="space-y-4">
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={handleFileChange}
                  disabled={isLoading}
                  className={`block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold transition-colors ${
                    theme === 'dark'
                      ? 'file:bg-blue-600 file:text-white file:hover:bg-blue-700 text-gray-300'
                      : 'file:bg-blue-500 file:text-white file:hover:bg-blue-600 text-gray-700'
                  } ${
                    isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  }`}
                />
                {selectedFile && (
                  <p
                    className={`text-sm ${
                      theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                    }`}
                  >
                    {t.currentFile} {selectedFile.name}
                  </p>
                )}
                {error && (
                  <p className="text-sm text-red-500">{error}</p>
                )}
              </div>
            </section>

            {/* Заключение */}
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
                {t.conclusion}
              </h2>
              <textarea
                value={conclusion}
                readOnly
                rows={15}
                className={`w-full p-4 rounded-xl border resize-none transition-colors ${
                  theme === 'dark'
                    ? 'bg-gray-900 border-gray-700 text-gray-100'
                    : 'bg-gray-50 border-gray-300 text-gray-900'
                }`}
                placeholder={t.conclusionPlaceholder}
              />
            </section>
          </div>

          {/* Правая колонка: изображение ЭКГ */}
          <div className="space-y-6">
            <section
              className={`p-6 rounded-2xl border shadow-lg h-full ${
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
                {t.ecgImage}
              </h2>
              <div className="h-[600px]">
                <ECGViewer imageUrl={ecgImageUrl} />
              </div>
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}

