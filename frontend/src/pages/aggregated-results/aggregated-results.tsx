import {
  FC, useState, useEffect, ReactElement,
} from 'react';
import { Row, Col } from 'react-grid-system';
import { Button, Table } from 'react-bootstrap';
import Loader from 'react-loader-spinner';

import { CaupoFrequency } from '../../services/http';
import RootPage from '../root/root';
import Box from '../../components/box/box';
import { ClustersService, ConsolidatedResult } from '../../services/clusters';

import 'react-loader-spinner/dist/loader/css/react-spinner-loader.css';

const AggregatedResultsPage: FC = () => {
  const [frequency, setFrequency] = useState('daily' as CaupoFrequency);
  const [consolidatedResults, setConsolidatedResults] = useState(null as ConsolidatedResult[] | null);

  const changeFrequency = (freq: CaupoFrequency) => {
    setFrequency(freq);
    setConsolidatedResults(null);
    ClustersService.getConsolidatedResults(freq)
      .then((res) => {
        setConsolidatedResults(res.data.data);
      }).catch(() => {});
  };

  const getResultsTable: () => ReactElement = () => {
    if (frequency === null || consolidatedResults === null) {
      return (
        <div className="center">
          <Loader
            type="Puff"
            color="#00BFFF"
            height={100}
            width={100}
          />
        </div>
      );
    }

    return (
      <>
        <Table striped bordered hover variant="dark">
          <thead>
            <tr>
              <th>Algorithm</th>
              <th>Embedder</th>
              <th>Avg. Sil. Score</th>
              <th>Valid Results</th>
            </tr>
          </thead>
          <tbody>
            {
              consolidatedResults.map((result) => (
                <tr key={`${frequency}-${result.embedder}-${result.algorithm}`}>
                  <td>{ result.algorithm }</td>
                  <td>{ result.embedder }</td>
                  <td>{ result.sil_score?.toFixed(7) }</td>
                  <td>{ result.valid_entries }</td>
                </tr>
              ))
            }
          </tbody>
        </Table>
      </>
    );
  };

  useEffect(() => {
    changeFrequency('daily');
  }, []);

  return (
    <RootPage title="Resultados Agregados">
      <Row>
        <Col md={12}>
          <Box>
            <>
              <p>
                En esta sección, puedes revisar los resultados de la clusterización o agrupamiento
                de tweets, acumulados y agregados a lo largo de las distintas entradas exitosas,
                como un resumen general de los resultados cuantitativos.
              </p>
              <h2>Frecuencia</h2>
              <Row>
                <Col md={3} offset={{ md: 3 }}>
                  <Button
                    variant={frequency === 'daily' ? 'success' : 'secondary'}
                    onClick={() => changeFrequency('daily')}
                  >
                    Diaria
                  </Button>
                </Col>
                <Col md={3}>
                  <Button
                    variant={frequency === 'weekly' ? 'success' : 'secondary'}
                    onClick={() => changeFrequency('weekly')}
                  >
                    Semanal
                  </Button>
                </Col>
              </Row>
              <br />
              <p>
                Actualmente está escogida la frecuencia&nbsp;
                {frequency === 'daily' ? 'diaria' : 'semanal'}
                .
              </p>
              <br />
              <h2>Resultados</h2>
              { getResultsTable() }
            </>
          </Box>
        </Col>
      </Row>
    </RootPage>
  );
};

export default AggregatedResultsPage;
