import { FC } from 'react';
import { BrowserRouter } from 'react-router-dom';

import AppRoutes from './routes/routes';

const App: FC = () => (
  <BrowserRouter>
    <AppRoutes />
  </BrowserRouter>
);

export default App;
