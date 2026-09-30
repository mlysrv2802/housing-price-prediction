import { Component } from '@angular/core';
import { HouseFormComponent } from './components/house-form/house-form.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [HouseFormComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'housing-price-ui';
}