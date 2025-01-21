import { CommonModule } from '@angular/common';
import { Component,ElementRef, HostListener, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-main',
  imports:[CommonModule],
  standalone: true,
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent implements OnInit {
  username: string | null=null;

  ngOnInit(): void{
    const user = localStorage.getItem('userLoggedIn'); // Adjust key as per your localStorage structure
    if (user) {
      const userObj = JSON.parse(user);
      this.username = userObj.username; // Assuming the stored object has a `username` field
    }
  }

  activeOption: number = 1; // Default selected option
  isDropdownVisible: boolean = false;

  constructor(private router: Router){}

  // Function to handle switching between options
  selectOption(option: number): void {
    this.activeOption = option;
  }

  // Function to toggle dropdown menu visibility
  toggleMenu(): void {
    this.isDropdownVisible = !this.isDropdownVisible
  }

  

  @HostListener('document:click',['$event'])
  disableMenu(event: MouseEvent): void {
    const targetElement = event.target as HTMLElement;
    if (!targetElement.closest('.top-right-menu')) {
      this.isDropdownVisible = false;
    }

  }

  login() : void{
    this.router.navigate(['/login'])
  }
  signup() : void{
    this.router.navigate(['/signup'])
  }
  logout(): void {
    localStorage.removeItem('userLoggedIn'); // Clear user data from local storage
    this.username = null; // Clear the username variable
    this.isDropdownVisible = false; // Close the dropdown
    window.location.reload();
  }

  selectedFileName: string | null = null; // Holds the name of the selected file
  errorMessage: string | null = null; // Holds the error message

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      if (file.type === 'application/pdf') {
        this.errorMessage = null; // Clear error message
        this.selectedFileName = file.name; // Set the file name
      } else {
        this.errorMessage = 'File must be a PDF.'; // Set error message
        this.selectedFileName = null; // Clear file name
      }
    }
  }

  triggerFileUpload(fileInput: HTMLInputElement): void {
    fileInput.click(); // Programmatically trigger the file input click
  }

  evaluateFile(): void {
    console.log('Evaluating file:', this.selectedFileName);
    // Add logic for file evaluation here
  }
}
