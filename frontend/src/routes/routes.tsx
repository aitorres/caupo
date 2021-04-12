import { FC } from 'react';
import { Route, Switch } from 'react-router-dom';

import HomePage from '../pages/home/home';

const AppRoutes: FC = () => (
  <Switch>
    <Route path="/" component={HomePage} />
  </Switch>
);

export default AppRoutes;
