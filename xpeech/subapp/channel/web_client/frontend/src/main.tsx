import React from 'react';
import ReactDOM from 'react-dom/client';
import { XProvider } from '@ant-design/x';
import '@ant-design/x-markdown/themes/light.css';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <XProvider locale={zhCN} theme={{ token: { colorPrimary: '#3370ff', borderRadius: 10 } }}>
      <App />
    </XProvider>
  </React.StrictMode>,
);
