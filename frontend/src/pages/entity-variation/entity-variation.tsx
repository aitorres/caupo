import { FC, useEffect, useState } from 'react';
import { Row, Col } from 'react-grid-system';
import Loader from 'react-loader-spinner';

import { EntitiesService, EntityVariationDataResponse } from '../../services/entities';

import RootPage from '../root/root';
import Box from '../../components/box/box';

import 'react-loader-spinner/dist/loader/css/react-spinner-loader.css';

const EntityVariationPage: FC = () => {
  const [dailyEntityVariationData, setDailyEntityVariationData] = useState(
    null as EntityVariationDataResponse | null,
  );
  const [weeklyEntityVariationData, setWeeklyEntityVariationData] = useState(
    null as EntityVariationDataResponse | null,
  );

  useEffect(() => {
    if (dailyEntityVariationData === null) {
      EntitiesService.getEntityVariationData('daily')
        .then((res) => {
          setDailyEntityVariationData(res.data);
        })
        .catch(() => {});
    }

    if (weeklyEntityVariationData === null) {
      EntitiesService.getEntityVariationData('weekly')
        .then((res) => {
          setWeeklyEntityVariationData(res.data);
        })
        .catch(() => {});
    }
  }, []);

  const loader: React.ReactElement = (
    <div className="center">
      <Loader
        type="Puff"
        color="#00BFFF"
        height={100}
        width={100}
      />
    </div>
  );

  const getDailyWordCloud: () => React.ReactElement = () => {
    if (dailyEntityVariationData === null) {
      return loader;
    }

    const tagElement = dailyEntityVariationData.data.map((tagData) => (
      <>
        <h3>{ tagData.tag }</h3>
        <Row>
          <Col md={3}>
            <h5>Nuevas entidades</h5>
            <ul>
              { tagData.persons.added.map((entity) => <li key={entity}>{entity}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Entidades eliminadas</h5>
            <ul>
              { tagData.persons.removed.map((entity) => <li key={entity}>{entity}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Nuevos hashtags</h5>
            <ul>
              { tagData.hashtags.added.map((hashtag) => <li key={hashtag}>{hashtag}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Hashtags eliminados</h5>
            <ul>
              { tagData.hashtags.removed.map((hashtag) => <li key={hashtag}>{hashtag}</li>) }
            </ul>
          </Col>
        </Row>
      </>
    ));

    return (
      <>
        { tagElement }
      </>
    );
  };

  const getWeeklyWordCloud: () => React.ReactElement = () => {
    if (weeklyEntityVariationData === null) {
      return loader;
    }

    const tagElement = weeklyEntityVariationData.data.map((tagData) => (
      <>
        <h3>{ tagData.tag }</h3>
        <Row>
          <Col md={3}>
            <h5>Nuevas entidades</h5>
            <ul>
              { tagData.persons.added.map((entity) => <li key={entity}>{entity}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Entidades eliminadas</h5>
            <ul>
              { tagData.persons.removed.map((entity) => <li key={entity}>{entity}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Nuevos hashtags</h5>
            <ul>
              { tagData.hashtags.added.map((hashtag) => <li key={hashtag}>{hashtag}</li>) }
            </ul>
          </Col>
          <Col md={3}>
            <h5>Hashtags eliminados</h5>
            <ul>
              { tagData.hashtags.removed.map((hashtag) => <li key={hashtag}>{hashtag}</li>) }
            </ul>
          </Col>
        </Row>
      </>
    ));

    return (
      <>
        { tagElement }
      </>
    );
  };

  return (
    <RootPage title="Variación de Entidades">
      <Row>
        <Col md={12}>
          <Box>
            <>
              <p>
                En esta sección, puedes revisar la variación entre las entidades (Named Entities) extraídas /
                reconocidas a partir de tweets recientes de política venezolana, con procesamiento de lenguaje
                natural.
              </p>
              <p>
                Las entidades se muestran en función de cada agrupamiento y en comparación al anterior / siguiente.
              </p>
              <h2>Análisis diario</h2>
              { getDailyWordCloud() }
              <h2>Análisis semanal</h2>
              { getWeeklyWordCloud() }
            </>
          </Box>
        </Col>
      </Row>
    </RootPage>
  );
};

export default EntityVariationPage;
