import {
  FC, useState, useEffect, ReactElement,
} from 'react';
import { Row, Col } from 'react-grid-system';
import { Button, Table } from 'react-bootstrap';

import { CaupoFrequency } from '../../services/http';
import RootPage from '../root/root';
import Box from '../../components/box/box';
import { ClustersService } from '../../services/clusters';

interface ResultsTable {
  [key: string]: {
    [key: string]: {
      [key: string]: number | null
    }
  }
}

const ClustersPage: FC = () => {
  const [frequency, setFrequency] = useState('daily' as CaupoFrequency);
  const [tags, setTags] = useState([] as string[]);
  const [selectedTag, setSelectedTag] = useState(null as string | null);
  const [algorithms, setAlgorithms] = useState([] as string[]);
  const [embedders, setEmbedders] = useState([] as string[]);
  const [resultsTable, setResultsTable] = useState({} as ResultsTable);

  const changeFrequency = (freq: CaupoFrequency) => {
    setTags([] as string[]);
    setSelectedTag(null);
    setFrequency(freq);
    ClustersService.getValidTags(freq)
      .then((res) => {
        setTags(res.data.data);
      })
      .catch(() => {
        setTags([] as string[]);
      });
  };

  const fetchAlgorithmNames = () => {
    ClustersService.getAlgorithmList()
      .then((res) => {
        setAlgorithms(res.data.data);
      }).catch(() => {});
  };

  const fetchEmbedderNames = () => {
    ClustersService.getEmbedderNames()
      .then((res) => {
        setEmbedders(res.data.data);
      }).catch(() => {});
  };

  useEffect(() => {
    changeFrequency('daily');
    fetchAlgorithmNames();
    fetchEmbedderNames();
  }, []);

  const getTagButtons: () => ReactElement[] = () => (
    tags.map((tag) => (
      <Col md={frequency === 'daily' ? 2 : 3} key={tag}>
        <Button variant={selectedTag === tag ? 'success' : 'secondary'} size="sm" onClick={() => setSelectedTag(tag)}>
          { tag }
        </Button>
      </Col>
    ))
  );

  const getResultsTable: () => ReactElement = () => {
    let highestSilScore = -2; // set to a non-valid value

    if (selectedTag === null) {
      return (
        <p>
          No se ha seleccionado una etiqueta válida.
        </p>
      );
    }

    if (!(selectedTag in resultsTable)) {
      resultsTable[selectedTag] = {};

      // filling table if needed
      algorithms.forEach((algorithm) => {
        if (!(algorithm in resultsTable[selectedTag])) {
          resultsTable[selectedTag][algorithm] = {};
        }

        embedders.forEach((embedder) => {
          Object.keys(resultsTable[selectedTag][algorithm]).forEach((key) => {
            const val = resultsTable[selectedTag][algorithm][key];

            if (val !== null && val >= highestSilScore) {
              highestSilScore = val;
            }
          });

          if (!(embedder in resultsTable[selectedTag][algorithm])) {
            resultsTable[selectedTag][algorithm][embedder] = null;
            ClustersService.getSilhouetteScore(
              frequency,
              selectedTag,
              algorithm,
              embedder,
            ).then((res) => {
              let value = res.data.data;
              if (value !== null) {
                value = parseFloat(value.toFixed(5));
              }
              resultsTable[selectedTag][algorithm][embedder] = value;
              setResultsTable(resultsTable);
            }).catch(() => {});
          }
        });
      });
    }

    return (
      <>
        <Table striped bordered hover variant="dark">
          <thead>
            <tr>
              <th>
                Embedder
              </th>
              { algorithms.map((algorithm) => (
                <th key={`${selectedTag}-${algorithm}`}>
                  { algorithm }
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            { embedders.map((embedder) => (
              <tr key={`${selectedTag}-${embedder}`}>
                <td>
                  {embedder}
                </td>
                { algorithms.map((algorithm) => (
                  <td
                    key={`${selectedTag}-${embedder}-${algorithm}`}
                    className={resultsTable[selectedTag][algorithm][embedder] === highestSilScore ? 'highest' : ''}
                  >
                    { resultsTable[selectedTag][algorithm][embedder] || '😢' }
                  </td>
                )) }
              </tr>
            )) }
          </tbody>
        </Table>
      </>
    );
  };

  return (
    <RootPage title="Clusters">
      <Row>
        <Col md={12}>
          <Box>
            <>
              <p>
                En esta sección, puedes revisar los resultados de la clusterización o agrupamiento
                de tweets, realizada utilizando distintos modelos de lenguaje (language embeddings)
                y algoritmos de clustering, combinados de distintas maneras.
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
                {frequency}
                &nbsp;con&nbsp;
                {tags.length}
                &nbsp;elementos (resultados) en el servidor.
              </p>
              { tags.length > 0 && (
                <>
                  <h2>Etiquetas de resultados</h2>
                  <Row>
                    { getTagButtons() }
                  </Row>
                  <br />
                  <p>
                    { selectedTag !== null && `Actualmente está seleccionada la etiqueta ${selectedTag}.` }
                    { selectedTag === null && 'Por favor, seleccione una etiqueta.' }
                  </p>
                  { selectedTag !== null && (
                    <>
                      <h2>Resultados</h2>
                      <p>
                        Se presenta la tabla de combinaciones de algoritmos y modelos de lenguaje para la etiqueta&nbsp;
                        {selectedTag}
                        .
                      </p>
                      { getResultsTable() }
                    </>
                  )}
                </>
              )}
            </>
          </Box>
        </Col>
      </Row>
    </RootPage>
  );
};

export default ClustersPage;
