import { AxiosPromise } from 'axios';
import HttpService from './http';

export type EntitiesFrequency = 'daily' | 'weekly' | 'monthly';

export interface EntitiesResponse {
  httpStatus: number,
  message: string,
  data: {
    tag: string,
    frequency: EntitiesFrequency,
    tweets_amount: number,
    entities: {
      all: {
        list: string[],
        set: string[],
        unique_amount: number,
      },
      organizations: {
        list: string[],
        set: string[],
        unique_amount: number,
      },
      persons: {
        list: string[],
        set: string[],
        unique_amount: number,
      },
      misc: {
        list: string[],
        set: string[],
        unique_amount: number,
      },
    },
    hashtags: {
      list: string[],
      set: string[],
      unique_amount: number,
    }
  }[]
}

export interface WordcloudResponse {
  httpStatus: number,
  message: string,
  data: string,
}

export interface EntityVariationDataResponse {
  httpStatus: number,
  message: string,
  data: {
    tag: string,
    frequency: string,
    tweets_amount: number,
    entities: {
      all: {
        added: string[],
        removed: string[],
      },
      organizations: {
        added: string[],
        removed: string[],
      },
      persons: {
        added: string[],
        removed: string[],
      },
      misc: {
        added: string[],
        removed: string[],
      },
    },
    hashtags: {
      added: string[],
      removed: string[],
    }
  }[]
}

export class EntitiesService {
  static baseUrl = 'https://api.caupo.xyz/entities';

  public static getEntities(frequency: EntitiesFrequency): AxiosPromise<EntitiesResponse> {
    const url = `${this.baseUrl}/get/${frequency}/1`;

    return HttpService.get(url) as AxiosPromise<EntitiesResponse>;
  }

  public static getWordcloud(frequency: EntitiesFrequency): AxiosPromise<WordcloudResponse> {
    const url = `${this.baseUrl}/wordcloud/get/${frequency}`;

    return HttpService.get(url) as AxiosPromise<WordcloudResponse>;
  }

  public static getEntityVariationData(frequency: EntitiesFrequency): AxiosPromise<EntityVariationDataResponse> {
    const url = `${this.baseUrl}/variations/get/${frequency}`;

    return HttpService.get(url) as AxiosPromise<EntityVariationDataResponse>;
  }
}
