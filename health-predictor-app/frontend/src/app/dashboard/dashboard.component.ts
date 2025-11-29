import { Component } from '@angular/core';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent {
  featureImportance = [
    { name: 'Mental Health Days', percentage: 13.67 },
    { name: 'BMI Value', percentage: 10.11 },
    { name: 'Income Categories', percentage: 9.39 },
    { name: 'Employment Status', percentage: 7.94 },
    { name: 'Diabetes Status', percentage: 7.44 },
    { name: 'Education Level', percentage: 5.95 },
    { name: 'Arthritis', percentage: 5.34 },
    { name: 'Age Group', percentage: 5.26 },
    { name: 'Personal Doctor', percentage: 5.19 },
    { name: 'Difficulty Doing Errands Alone', percentage: 4.92 },
    { name: 'Exercise Past 30 Days', percentage: 4.65 },
    { name: 'Difficulty Concentrating', percentage: 4.09 },
    { name: 'Could Not Afford Doctor', percentage: 3.70 },
    { name: 'Primary Insurance', percentage: 3.46 },
    { name: 'Coronary Heart Disease', percentage: 3.20 },
    { name: 'Difficulty Dressing/Bathing', percentage: 3.18 },
    { name: 'Sex', percentage: 2.52 }
  ];
}
