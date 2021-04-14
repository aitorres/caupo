import React, { FC } from 'react';
import PropTypes from 'prop-types';
import { Container, Row, Col } from 'react-grid-system';

import Header from '../../components/header/header';
import Sidebar from '../../components/sidebar/sidebar';

import '@fontsource/roboto';
import '../../assets/css/general.css';

interface RootPageProps {
  title: string,
  children: React.ReactChild
}

const getCaupoQuote: () => string = () => {
  const lines: string[] = [
    'Ojo de barro y Water de Urgencia.',
    'Si en vez de dormir bailara tango con sus ministros...',
    'Claro que uno está cansado y quiere un poco de diversión',
    'Cacique Ojo de Perla',
    'Si adora la vaca, ¡duerme!',
    'Si al becerro adora, ¡duerme!',
    '¿Hasta cuándo duerme usted, señor Presidente?',
  ];

  return lines[Math.floor(Math.random() * lines.length)];
};

const RootPage: FC<RootPageProps> = ({ title, children }) => (
  <div className="page">
    <Header title={title} />
    <div className="content">
      <Container>
        <Row>
          <Col md={3}>
            <Sidebar />
          </Col>
          <Col md={9}>
            { children }
          </Col>
        </Row>
      </Container>
    </div>
    <footer>
      {getCaupoQuote()}
    </footer>
  </div>
);
RootPage.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.element.isRequired,
};

export default RootPage;
