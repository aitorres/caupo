import { FC, useEffect, useState } from 'react';
import { Row, Col } from 'react-grid-system';
import Loader from 'react-loader-spinner';

import { EntitiesService } from '../../services/entities';

import RootPage from '../root/root';
import Box from '../../components/box/box';

import 'react-loader-spinner/dist/loader/css/react-spinner-loader.css';

const EntitiesPage: FC = () => {
  const [dailyEntities, setDailyEntities] = useState('');
  const [weeklyEntities, setWeeklyEntities] = useState('');
  const [monthlyEntities, setMonthlyEntities] = useState('');

  useEffect(() => {
    if (dailyEntities === '') {
      EntitiesService.getWordcloud('daily')
        .then((res) => {
          setDailyEntities(res.data.data);
        })
        .catch(() => {});
    }

    if (weeklyEntities === '') {
      EntitiesService.getWordcloud('weekly')
        .then((res) => {
          setWeeklyEntities(res.data.data);
        })
        .catch(() => {});
    }

    if (monthlyEntities === '') {
      EntitiesService.getWordcloud('monthly')
        .then((res) => {
          setMonthlyEntities(res.data.data);
        })
        .catch(() => {});
    }
  });

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
    if (dailyEntities === '') {
      return (
        <>
          <p>Loading daily entities...</p>
          { loader }
        </>
      );
    }

    return (
      <p>
        <img className="center" src={`data:image/png;base64,${dailyEntities}`} alt="Wordcloud" />
      </p>
    );
  };

  const getWeeklyWordCloud: () => React.ReactElement = () => {
    if (weeklyEntities === '') {
      return (
        <>
          <p>Loading weekly entities...</p>
          { loader }
        </>
      );
    }

    return (
      <p>
        <img className="center" src={`data:image/png;base64,${weeklyEntities}`} alt="Wordcloud" />
      </p>
    );
  };

  const getMonthlyWordCloud: () => React.ReactElement = () => {
    if (monthlyEntities === '') {
      return (
        <>
          <p>Loading monthly entities...</p>
          { loader }
        </>
      );
    }

    return (
      <p>
        <img className="center" src={`data:image/png;base64,${monthlyEntities}`} alt="Wordcloud" />
      </p>
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
