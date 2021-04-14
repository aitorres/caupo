import { FC } from 'react';
import { Route, Switch } from 'react-router-dom';

import HomePage from '../pages/home/home';
import EntitiesPage from '../pages/entities/entities';

const AppRoutes: FC = () => (
  <Switch>
    <Route path="/entities" component={EntitiesPage} />
    <Route path="/" component={HomePage} />
  </Switch>
);

export default AppRoutes;
