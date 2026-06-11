import React from 'react';
import ReactDOM from 'react-dom/client';
import { App as AntApp, ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { XProvider } from '@ant-design/x';
import 'antd/dist/reset.css';
import './styles.css';
import { DesktopApp } from './DesktopApp';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider locale={zhCN} theme={{ token: { borderRadius: 6 } }}>
      <AntApp>
        <XProvider>
          <DesktopApp />
        </XProvider>
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>,
);
