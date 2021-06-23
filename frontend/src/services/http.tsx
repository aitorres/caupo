import axios, { AxiosPromise } from 'axios';

export type CaupoFrequency = 'daily' | 'weekly' | 'monthly';

export interface BaseResponse {
  httpStatus: number,
  message: string,
  data: unknown
}

export class HttpService {
  public static get(url: string): AxiosPromise<unknown> {
    const xhr = axios.get(url);

    return xhr;
  }
}
