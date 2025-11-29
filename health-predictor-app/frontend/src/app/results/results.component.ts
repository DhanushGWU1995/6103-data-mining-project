import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-results',
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.css']
})
export class ResultsComponent {
  @Input() result: any;

  retakeAssessment() {
    // This would need to be implemented to reset the form
    window.location.reload();
  }

  shareResults() {
    // Simple share functionality
    if (navigator.share) {
      navigator.share({
        title: 'My Health Assessment Results',
        text: `My health risk assessment shows: ${this.result.prediction} with ${this.result.probability * 100}% probability.`,
        url: window.location.href
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(`My health risk assessment: ${this.result.prediction} (${this.result.probability * 100}% probability)`);
      alert('Results copied to clipboard!');
    }
  }
}
