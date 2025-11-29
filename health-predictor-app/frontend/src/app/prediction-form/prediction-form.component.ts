import { Component, Output, EventEmitter } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';

@Component({
  selector: 'app-prediction-form',
  templateUrl: './prediction-form.component.html',
  styleUrls: ['./prediction-form.component.css']
})
export class PredictionFormComponent {
  @Output() predictionResult = new EventEmitter<any>();
  form: FormGroup;
  currentPanel = 0;
  totalPanels = 4;
  questions = [
    {
      field: 'Income_Categories',
      question: 'What is your income category?',
      type: 'select',
      options: [
        { value: 1, label: 'Less than $10,000' },
        { value: 2, label: 'Less than $15,000 ($10,000 to < $15,000)' },
        { value: 3, label: 'Less than $20,000 ($15,000 to < $20,000)' },
        { value: 4, label: 'Less than $25,000 ($20,000 to < $25,000)' },
        { value: 5, label: 'Less than $35,000 ($25,000 to < $35,000)' },
        { value: 6, label: 'Less than $50,000 ($35,000 to < $50,000)' },
        { value: 7, label: 'Less than $75,000 ($50,000 to < $75,000)' },
        { value: 8, label: 'Less than $100,000 ($75,000 to < $100,000)' },
        { value: 9, label: 'Less than $150,000 ($100,000 to < $150,000)' },
        { value: 10, label: 'Less than $200,000 ($150,000 to < $200,000)' },
        { value: 11, label: '$200,000 or more' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Could_Not_Afford_Doctor',
      question: 'Could you not afford to see a doctor in the past year? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Employment_Status',
      question: 'What is your employment status?',
      type: 'select',
      options: [
        { value: 1, label: 'Employed for wages' },
        { value: 2, label: 'Self-employed' },
        { value: 3, label: 'Out of work for 1 year or more' },
        { value: 4, label: 'Out of work for less than 1 year' },
        { value: 5, label: 'A homemaker' },
        { value: 6, label: 'A student' },
        { value: 7, label: 'Retired' },
        { value: 8, label: 'Unable to work' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Primary_Insurance',
      question: 'What is your primary insurance type?',
      type: 'select',
      options: [
        { value: 1, label: 'A plan purchased through an employer or union (including plans purchased through another person’s employer)' },
        { value: 2, label: 'A private nongovernmental plan that you or another family member buys on your own' },
        { value: 3, label: 'Medicare' },
        { value: 4, label: 'Medigap' },
        { value: 5, label: 'Medicaid' },
        { value: 6, label: 'Children’s Health Insurance Program (CHIP)' },
        { value: 7, label: 'Military related health care: TRICARE (CHAMPUS) / VA health care / CHAMP-VA' },
        { value: 8, label: 'Indian Health Service' },
        { value: 9, label: 'State sponsored health plan' },
        { value: 10, label: 'Other government program' },
        { value: 88, label: 'No coverage of any type' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Education_Level',
      question: 'What is your education level?',
      type: 'select',
      options: [
        { value: 1, label: 'Never attended school or only kindergarten' },
        { value: 2, label: 'Grades 1 through 8 (Elementary)' },
        { value: 3, label: 'Grades 9 through 11 (Some high school)' },
        { value: 4, label: 'Grade 12 or GED (High school graduate)' },
        { value: 5, label: 'College 1 year to 3 years (Some college or technical school)' },
        { value: 6, label: 'College 4 years or more (College graduate)' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Age_Group_5yr',
      question: 'What is your age group?',
      type: 'select',
      options: [
        { value: 1, label: 'Age 18 to 24' },
        { value: 2, label: 'Age 25 to 29' },
        { value: 3, label: 'Age 30 to 34' },
        { value: 4, label: 'Age 35 to 39' },
        { value: 5, label: 'Age 40 to 44' },
        { value: 6, label: 'Age 45 to 49' },
        { value: 7, label: 'Age 50 to 54' },
        { value: 8, label: 'Age 55 to 59' },
        { value: 9, label: 'Age 60 to 64' },
        { value: 10, label: 'Age 65 to 69' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Sex',
      question: 'What is your sex? (1 = Male, 2 = Female)',
      type: 'select',
      options: [
        { value: 1, label: 'Male' },
        { value: 2, label: 'Female' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'BMI_Value',
      question: 'What is your BMI value?',
      type: 'number',
      min: 10.0,
      max: 60.0,
      errorMessage: 'Please enter a valid BMI between 10.0 and 60.0.'
    },
    {
      field: 'Exercise_Past_30_Days',
      question: 'Have you exercised in the past 30 days? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Mental_Health_Days',
      question: 'How many days in the past 30 have you had poor mental health?',
      type: 'number',
      min: 0,
      max: 30,
      errorMessage: 'Please enter a value between 0 and 30.'
    },
    {
      field: 'Diabetes_Status',
      question: 'Do you have diabetes?',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'Yes, but female told only during pregnancy' },
        { value: 3, label: 'No' },
        { value: 4, label: 'No, pre-diabetes or borderline diabetes' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Coronary_Heart_Disease',
      question: 'Do you have coronary heart disease? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Personal_Doctor',
      question: 'Do you have a personal doctor? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Difficulty_Doing_Errands_Alone',
      question: 'Do you have difficulty doing errands alone? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Difficulty_Dressing_Bathing',
      question: 'Do you have difficulty dressing or bathing? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Difficulty_Concentrating',
      question: 'Do you have difficulty concentrating? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    },
    {
      field: 'Arthritis',
      question: 'Do you have arthritis? (1 = Yes, 2 = No)',
      type: 'select',
      options: [
        { value: 1, label: 'Yes' },
        { value: 2, label: 'No' }
      ],
      errorMessage: 'Please select an option.'
    }
  ];

  constructor(private fb: FormBuilder) {
    this.form = this.fb.group({});
    this.questions.forEach(q => {
      const validators = [Validators.required];
      if (q.type === 'number') {
        validators.push(Validators.min(q.min), Validators.max(q.max));
      }
      this.form.addControl(q.field, this.fb.control('', validators));
    });
  }

  nextPanel() {
    if (this.currentPanel < this.totalPanels - 1 && this.isCurrentPanelValid()) {
      this.currentPanel++;
    }
  }

  previousPanel() {
    if (this.currentPanel > 0) {
      this.currentPanel--;
    }
  }

  isCurrentPanelValid(): boolean {
    let panelQuestions: any[] = [];

    // Match the slicing logic from the HTML template
    switch (this.currentPanel) {
      case 0:
        panelQuestions = this.questions.slice(0, 5); // Financial Factors: 5 questions
        break;
      case 1:
        panelQuestions = this.questions.slice(5, 7); // Demographic: 2 questions
        break;
      case 2:
        panelQuestions = this.questions.slice(7, 10); // Health Metrics: 3 questions
        break;
      case 3:
        panelQuestions = this.questions.slice(10); // Health Conditions: 6 questions
        break;
      default:
        return false;
    }

    return panelQuestions.every(question => {
      const control = this.form.get(question.field);
      return control && control.valid;
    });
  }

  submit() {
    this.predictionResult.emit(this.form.value);
  }
}
