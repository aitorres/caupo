import { FC, useEffect, useState } from 'react';
import { Row, Col } from 'react-grid-system';
import Loader from 'react-loader-spinner';

import { EntitiesService } from '../../services/entities';

import RootPage from '../root/root';
import Box from '../../components/box/box';

import 'react-loader-spinner/dist/loader/css/react-spinner-loader.css';

const EntitiesPage: FC = () => {
  const [dailyEntities, setDailyEntities] = useState(null as unknown);
  const [weeklyEntities, setWeeklyEntities] = useState(null as unknown);
  const [monthlyEntities, setMonthlyEntities] = useState(null as unknown);

  useEffect(() => {
    EntitiesService.getEntities('daily')
      .then((res) => {
        setDailyEntities(res.data);
      })
      .catch(() => {});

    EntitiesService.getEntities('weekly')
      .then((res) => {
        setWeeklyEntities(res.data);
      })
      .catch(() => {});

    EntitiesService.getEntities('monthly')
      .then((res) => {
        setMonthlyEntities(res.data);
      })
      .catch(() => {});
  });

  const getDailyWordCloud: () => React.ReactElement = () => {
    if (dailyEntities === null) {
      return (
        <>
          <p>Loading daily entities...</p>
          <div className="center">
            <Loader
              type="Puff"
              color="#00BFFF"
              height={100}
              width={100}
            />
          </div>
        </>
      );
    }

    return (
      <p>Loaded (waiting for wordcloud render)</p>
    );
  };

  const getWeeklyWordCloud: () => React.ReactElement = () => {
    if (weeklyEntities === null) {
      return (
        <>
          <p>Loading weekly entities...</p>
          <div className="center">
            <Loader
              type="Puff"
              color="#00BFFF"
              height={100}
              width={100}
            />
          </div>
        </>
      );
    }

    return (
      <p>Loaded (waiting for wordcloud render)</p>
    );
  };

  const getMonthlyWordCloud: () => React.ReactElement = () => {
    if (monthlyEntities === null) {
      return (
        <>
          <p>Loading monthly entities...</p>
          <div className="center">
            <Loader
              type="Puff"
              color="#00BFFF"
              height={100}
              width={100}
            />
          </div>
        </>
      );
    }

    return (
      <p>Loaded (waiting for wordcloud render)</p>
    );
  };

  return (
    <RootPage title="Entidades">
      <Row>
        <Col md={12}>
          <Box>
            <>
              <p>
                En esta sección, puedes revisar las entidades (Named Entities) extraídas / reconocidas a partir de
                tweets recientes de política venezolana, utilizando modelos de inteligencia artificial y técnicas de
                procesamiento de lenguaje natural.
              </p>
              <p>
                Las entidades se visualizan utilizando nubes de palabras, en donde el
                tamaño de cada palabra / entidad / hashtag corresponde a qué tanto es usado ese término en el conjunto
                de tweets analizado.
              </p>
              <h2>Análisis diario</h2>
              { getDailyWordCloud() }
              <h2>Análisis semanal</h2>
              { getWeeklyWordCloud() }
              <h2>Análisis mensual</h2>
              { getMonthlyWordCloud() }
            </>
          </Box>
        </Col>
      </Row>
    </RootPage>
  );
};

export default EntitiesPage;
