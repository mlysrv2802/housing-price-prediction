import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

// This mirrors HouseFeatures in api.py — keep both in sync.
export interface HouseFeatures {
  area_sqft: number;
  bedrooms: number;
  bathrooms: number;
  age_years: number;
  distance_to_city_km: number;
  location_tier: string;
}

export interface PredictionResponse {
  predicted_price: number;
}

@Injectable({
  providedIn: 'root',
})
export class PredictionService {
  // In a real app this goes in environment.ts (environment.apiUrl),
  // not hardcoded here — same idea as an application.properties value.
  private readonly apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  predictPrice(features: HouseFeatures): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`${this.apiUrl}/predict`, features);
  }
}
