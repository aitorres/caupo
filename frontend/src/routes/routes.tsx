import { FC } from 'react';
import { Route, Switch } from 'react-router-dom';

import HomePage from '../pages/home/home';
import EntitiesPage from '../pages/entities/entities';
import EntityVariationPage from '../pages/entity-variation/entity-variation';
import ClustersPage from '../pages/clusters/clusters';
import ClusterResultPage from '../pages/cluster-result/cluster-result';
import AggregatedResultsPage from '../pages/aggregated-results/aggregated-results';

const AppRoutes: FC = () => (
  <Switch>
    <Route path="/aggregated-results" component={AggregatedResultsPage} />
    <Route path="/cluster-results" component={ClusterResultPage} />
    <Route path="/clusters" component={ClustersPage} />
    <Route path="/entity-variation" component={EntityVariationPage} />
    <Route path="/entities" component={EntitiesPage} />
    <Route path="/" component={HomePage} />
  </Switch>
);

export default AppRoutes;
