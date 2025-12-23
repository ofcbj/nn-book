import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import ko from './locales/ko.json';
import ja from './locales/ja.json';

i18n
  .use(initReactI18next)
  .init({
    resources: {
      ko: { translation: ko },
      ja: { translation: ja }
    },
    lng: 'ja', // 기본 언어를 일본어로 변경
    fallbackLng: 'ko',
    interpolation: {
      escapeValue: false // React는 XSS를 자동으로 escape함
    }
  });

export default i18n;
