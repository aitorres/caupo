import { FC } from 'react';
import { Row, Col } from 'react-grid-system';

import RootPage from '../root/root';

import '../../assets/css/general.css';

const Home: FC = () => (
  <RootPage title="Home">
    <Row className="center">
      <Col md={6}>
        <p>
          Hola
        </p>
      </Col>
    </Row>
  </RootPage>
);

export default Home;
