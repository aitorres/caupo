import {
  FC, useState, useEffect, ReactElement,
} from 'react';
import { Row, Col } from 'react-grid-system';
import { Button, Table } from 'react-bootstrap';
import Loader from 'react-loader-spinner';

import { CaupoFrequency } from '../../services/http';
import RootPage from '../root/root';
import Box from '../../components/box/box';
import { ClustersService, ResultListResponse, Result } from '../../services/clusters';

import 'react-loader-spinner/dist/loader/css/react-spinner-loader.css';

const ClusterResultPage: FC = () => {
  const [frequency, setFrequency] = useState('daily' as CaupoFrequency);
  const [tags, setTags] = useState([] as string[]);
  const [selectedTag, setSelectedTag] = useState(null as string | null);
  const [results, setResults] = useState(null as ResultListResponse | null);
  const [selectedResult, setSelectedResult] = useState(null as {
    embedder: string,
    algorithm: string
  } | null);

  const changeFrequency = (freq: CaupoFrequency) => {
    setTags([] as string[]);
    setSelectedTag(null);
    setSelectedResult(null);
    setFrequency(freq);
    ClustersService.getValidTags(freq)
      .then((res) => {
        setTags(res.data.data);
      })
      .catch(() => {
        setTags([] as string[]);
      });
  };

  const changeSelectedTag = (tag: string) => {
    setSelectedTag(tag);
    setSelectedResult(null);
    ClustersService.getResultList(frequency, tag)
      .then((res) => setResults(res.data))
      .catch(() => {});
  };

  const getResultDetails: () => ReactElement = () => {
    if (selectedTag === null || selectedResult === null) {
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

    const baseUrl = `https://static.caupo.xyz/cluster_tags/${frequency}`;
    const imgurl = `${baseUrl}/${selectedResult.embedder}/${selectedResult.algorithm}/${selectedTag}_plot.png`;

    return (
      <>
        <img
          src={imgurl}
          alt="results"
          className="center"
        />
      </>
    );
  };

  const getResultsTable: () => ReactElement = () => {
    if (selectedTag === null || results === null) {
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
              <th />
              <th>Embedder</th>
              <th>Algorithm</th>
              <th>Sil. Score</th>
              <th># Clusters </th>
              <th>Cluster Size</th>
              <th>Clustered Tweets</th>
              <th>Ignored Tweets</th>
            </tr>
          </thead>
          <tbody>
            {
              results.data && results.data.map((result) => (
                <tr key={`${selectedTag}-${result.embedder}-${result.algorithm}`}>
                  <td>
                    <input
                      type="radio"
                      name="selectedResult"
                      aria-label="test"
                      onChange={() => setSelectedResult({ embedder: result.embedder, algorithm: result.algorithm })}
                    />
                  </td>
                  <td>{ result.embedder }</td>
                  <td>{ result.algorithm }</td>
                  <td>{ result.scores.silhouette?.toFixed(7) }</td>
                  <td>{ Object.keys(result.averageSentiment).length }</td>
                  <td>
                    min:&nbsp;
                    { result.minClusterSize }
                    <br />
                    avg&nbsp;
                    { result.avgClusterSize }
                    <br />
                    max&nbsp;
                    { result.maxClusterSize }
                  </td>
                  <td>{ result.validTweetsAmount }</td>
                  <td>{ result.tweetsAmount - result.validTweetsAmount }</td>
                </tr>
              ))
            }
          </tbody>
        </Table>
      </>
    );
  };

  const getTopicsTable: () => ReactElement = () => {
    if (selectedResult === null || results === null) {
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

    let resultData: Result | null = null;
    results.data.forEach((result) => {
      if (result.algorithm === selectedResult.algorithm && result.embedder === selectedResult.embedder) {
        resultData = result;
      }
    });

    if (resultData === null) {
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
    resultData = resultData as Result;

    const clusterKeys = Object.keys(resultData.averageSentiment);

    return (
      <>
        <Table striped bordered hover variant="dark">
          <thead>
            <tr>
              <th>Cluster</th>
              <th>Topics</th>
              <th>Average Sentiment</th>
            </tr>
          </thead>
          <tbody>
            {
              clusterKeys.map((clusterKey) => (
                <tr key={clusterKey}>
                  <td>{ clusterKey }</td>
                  <td>
                    { resultData?.clusterThemes[clusterKey].map((theme) => (
                      <>
                        { theme }
                        <br />
                      </>
                    )) }
                  </td>
                  <td>{ resultData?.averageSentiment[clusterKey].toFixed(5) }</td>
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

  const getTagButtons: () => ReactElement[] = () => (
    tags.map((tag) => (
      <Col md={frequency === 'daily' ? 2 : 3} key={tag}>
        <Button
          variant={selectedTag === tag ? 'success' : 'secondary'}
          onClick={() => changeSelectedTag(tag)}
          size="sm"
        >
          { tag }
        </Button>
      </Col>
    ))
  );

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
                {frequency === 'daily' ? 'diaria' : 'semanal'}
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
                      { selectedResult !== null && getResultDetails() }
                      { selectedResult !== null && getTopicsTable() }
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

export default ClusterResultPage;
