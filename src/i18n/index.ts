import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import ko from "./locales/ko.json";
import ja from "./locales/ja.json";
import en from "./locales/en.json";

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    ko: { translation: ko },
    ja: { translation: ja },
  },
  lng: "en", // Default language set to English
  fallbackLng: "en",
  interpolation: {
    escapeValue: false, // React automatically escapes XSS
  },
});

export default i18n;
