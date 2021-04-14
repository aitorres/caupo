import { FC } from 'react';
import { Row, Col } from 'react-grid-system';

import RootPage from '../root/root';
import Box from '../../components/box/box';

const EntitiesPage: FC = () => (
  <RootPage title="Entidades">
    <Row>
      <Col md={12}>
        <Box>
          <>
            <p>
              En esta sección, puedes revisar las entidades (Named Entities) extraídas / reconocidas a partir de tweets
              recientes de política venezolana, utilizando modelos de inteligencia artificial y técnicas de
              procesamiento de lenguaje natural.
            </p>
            <p>
              Las entidades se visualizan utilizando nubes de palabras, en donde el
              tamaño de cada palabra / entidad / hashtag corresponde a qué tanto es usado ese término en el conjunto
              de tweets analizado.
            </p>
          </>
        </Box>
      </Col>
    </Row>
  </RootPage>
);

export default EntitiesPage;
