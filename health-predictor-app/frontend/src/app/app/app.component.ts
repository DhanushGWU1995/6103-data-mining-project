import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  activeTab = 'dashboard';
  result: any = null;

  constructor(private http: HttpClient) {}

  onPrediction(data: any) {
    this.http.post('/api/predict', data).subscribe(result => {
      this.result = result;
      this.activeTab = 'results';
    });
  }
}
