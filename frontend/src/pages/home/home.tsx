import { FC } from 'react';
import { Row, Col } from 'react-grid-system';

import RootPage from '../root/root';
import Box from '../../components/box/box';

const HomePage: FC = () => (
  <RootPage title="Inicio">
    <Row>
      <Col md={12}>
        <Box>
          <>
            <p>
              Hola, este proyecto está en (perenne) construcción.
            </p>
            <p>
              CAUPO (Cluster Analysis of Unsupervised Political Opinions) es un proyecto de investigación realizado
              realizado como parte de mi proyecto de grado para optar al título de Ingeniero en Computación en la
              Universidad Simón Bolívar.
            </p>
            <p>
              Las citas del pié de página son tomadas y (mal)formateadas de
              <em>
                &ldquo;¿Duerme usted, señor Presidente?&rdquo;,&nbsp;
              </em>
              poema largo de Caupolicán Ovalles, poeta venezolano del siglo XX.
            </p>
          </>
        </Box>
      </Col>
    </Row>
  </RootPage>
);

export default HomePage;
