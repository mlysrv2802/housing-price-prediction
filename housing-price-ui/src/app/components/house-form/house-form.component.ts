import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, Validators } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-house-form',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './house-form.component.html',
  styleUrl: './house-form.component.scss'
})
export class HouseFormComponent {
  private fb = inject(FormBuilder);
  private http = inject(HttpClient);

  // 1. Convert component state to Signals
  predictedPrice = signal<number | null>(null);
  errorMessage = signal<string | null>(null);
  isLoading = signal<boolean>(false);

  locationTiers = ['Tier1', 'Tier2', 'Tier3'];

  form = this.fb.group({
    area_sqft: [1400, [Validators.required, Validators.min(1)]],
    bedrooms: [3, [Validators.required, Validators.min(0)]],
    bathrooms: [2, [Validators.required, Validators.min(0)]],
    age_years: [5, [Validators.required, Validators.min(0)]],
    distance_to_city_km: [6.2, [Validators.required, Validators.min(0)]],
    location_tier: ['Tier2', Validators.required]
  });

  onSubmit() {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    // Reset error & set loading via signal setters
    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.predictedPrice.set(null);

    this.http.post<{ predicted_price: number }>(
      'http://localhost:8000/predict',
      this.form.value
    ).subscribe({
      next: (response) => {
        // 2. Updating signal values triggers immediate UI re-rendering
        this.predictedPrice.set(response.predicted_price);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.errorMessage.set(err.error?.detail || 'Failed to predict price.');
        this.isLoading.set(false);
      }
    });
  }
}