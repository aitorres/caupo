import { AxiosPromise } from 'axios';
import { CaupoFrequency, HttpService, BaseResponse } from './http';

export interface AlgorithmListResponse extends BaseResponse {
  data: string[]
}

export interface EmbedderNamesResponse extends BaseResponse {
  data: string[]
}

export interface ValidTagsResponse extends BaseResponse {
  data: string[]
}

export interface SilhouetteScoreResponse extends BaseResponse {
  data: number | null
}

export interface Result {
  frequency: string,
  tag: string,
  algorithm: string,
  embedder: string,
  clusterThemes: {
    [key: string]: string[]
  },
  time: number,
  averageSentiment: {
    [key: string]: number
  },
  scores: {
    silhouette: number | null,
    davies_bouldin: number | null,
    calinski_harabasz: number | null
  },
  tweetsAmount: number,
  validTweetsAmount: number,
  minClusterSize: number,
  avgClusterSize: number,
  maxClusterSize: number,
}

export interface ResultListResponse extends BaseResponse {
  data: Result[]
}

export interface ConsolidatedResult {
  algorithm: string,
  embedder: string,
  valid_entries: number,
  sil_score: number | null,
  weighted_score: number | null,
}

export interface ConsolidatedResultsResponse extends BaseResponse {
  data: ConsolidatedResult[]
}

export class ClustersService {
  static baseUrl = 'https://api.caupo.xyz/clusters';

  public static getEmbedderNames(): AxiosPromise<EmbedderNamesResponse> {
    const url = `${this.baseUrl}/embedders/list`;

    return HttpService.get(url) as AxiosPromise<EmbedderNamesResponse>;
  }

  public static getAlgorithmList(): AxiosPromise<AlgorithmListResponse> {
    const url = `${this.baseUrl}/clustering-algorithms/list`;

    return HttpService.get(url) as AxiosPromise<AlgorithmListResponse>;
  }

  public static getValidTags(frequency: CaupoFrequency): AxiosPromise<ValidTagsResponse> {
    const url = `${this.baseUrl}/tags/list/${frequency}`;

    return HttpService.get(url) as AxiosPromise<ValidTagsResponse>;
  }

  public static getSilhouetteScore(
    frequency: CaupoFrequency,
    tag: string,
    algorithm: string,
    embedder: string,
  ): AxiosPromise<SilhouetteScoreResponse> {
    const url = `${this.baseUrl}/results/silhouette`;

    return HttpService.post(
      url,
      {
        frequency,
        tag,
        algorithm,
        embedder,
      },
    ) as AxiosPromise<SilhouetteScoreResponse>;
  }

  public static getResultList(
    frequency: CaupoFrequency,
    tag: string,
  ): AxiosPromise<ResultListResponse> {
    const url = `${this.baseUrl}/results/list`;

    return HttpService.post(
      url,
      {
        frequency,
        tag,
      },
    ) as AxiosPromise<ResultListResponse>;
  }

  public static getConsolidatedResults(frequency: string): AxiosPromise<ConsolidatedResultsResponse> {
    const url = `${this.baseUrl}/results/consolidated/${frequency}`;

    return HttpService.get(url) as AxiosPromise<ConsolidatedResultsResponse>;
  }
}
