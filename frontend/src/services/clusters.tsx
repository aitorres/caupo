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
}