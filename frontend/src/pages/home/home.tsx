import { FC } from 'react';
import { Row, Col } from 'react-grid-system';

import RootPage from '../root/root';
import Box from '../../components/box/box';

import '../../assets/css/general.css';

const Home: FC = () => (
  <RootPage title="Inicio">
    <Row>
      <Col md={12}>
        <Box>
          <>
            Hola, este proyecto está en (perenne) construcción.
          </>
        </Box>
      </Col>
    </Row>
  </RootPage>
);

export default Home;
