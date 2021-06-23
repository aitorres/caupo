import { AxiosPromise } from 'axios';
import { HttpService, CaupoFrequency, BaseResponse } from './http';

export interface EntitiesResponse extends BaseResponse {
  data: {
    tag: string,
    frequency: CaupoFrequency,
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

export interface WordcloudResponse extends BaseResponse {
  data: string,
}

export interface EntityVariationDataResponse extends BaseResponse {
  data: {
    tag: string,
    frequency: string,
    tweets_amount: number,
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
    hashtags: {
      added: string[],
      removed: string[],
    }
  }[]
}

export class EntitiesService {
  static baseUrl = 'https://api.caupo.xyz/entities';

  public static getEntities(frequency: CaupoFrequency): AxiosPromise<EntitiesResponse> {
    const url = `${this.baseUrl}/get/${frequency}/1`;

    return HttpService.get(url) as AxiosPromise<EntitiesResponse>;
  }

  public static getWordcloud(frequency: CaupoFrequency): AxiosPromise<WordcloudResponse> {
    const url = `${this.baseUrl}/wordcloud/get/${frequency}`;

    return HttpService.get(url) as AxiosPromise<WordcloudResponse>;
  }

  public static getEntityVariationData(frequency: CaupoFrequency): AxiosPromise<EntityVariationDataResponse> {
    const url = `${this.baseUrl}/variations/get/${frequency}`;

    return HttpService.get(url) as AxiosPromise<EntityVariationDataResponse>;
  }
}
