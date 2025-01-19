import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

@Component({
  selector: 'app-main',
  imports:[CommonModule],
  standalone: true,
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent {
  activeOption: number = 1; // Default selected option
  isDropdownVisible: boolean = false;

  // Function to handle switching between options
  selectOption(option: number): void {
    this.activeOption = option;
  }

  // Function to toggle dropdown menu visibility
  toggleMenu(): void {
    this.isDropdownVisible = !this.isDropdownVisible;
  }
}
