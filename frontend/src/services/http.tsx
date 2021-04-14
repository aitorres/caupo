import axios, { AxiosPromise } from 'axios';

class HttpService {
  public static get(url: string): AxiosPromise<unknown> {
    const xhr = axios.get(url);

    return xhr;
  }
}

export default HttpService;
