import React, { FC } from 'react';
import PropTypes from 'prop-types';
import { Container, Row, Col } from 'react-grid-system';

import Header from '../../components/header/header';

import '../../assets/css/general.css';

interface RootPageProps {
  title: string,
  children: React.ReactChild
}

const RootPage: FC<RootPageProps> = ({ title, children }) => (
  <div className="page">
    <header className="Header">
      <Header title={title} />
    </header>
    <div className="content">
      <Container>
        <Row>
          <Col md={3}>
            Sidebar
          </Col>
          <Col md={9}>
            { children }
          </Col>
        </Row>
      </Container>
    </div>
  </div>
);
RootPage.propTypes = {
  title: PropTypes.string.isRequired,
  children: PropTypes.element.isRequired,
};

export default RootPage;
