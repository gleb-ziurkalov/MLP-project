import { CommonModule } from '@angular/common';
import { Component,ElementRef, HostListener, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { Eval } from '../models/eval';
import { UserService } from '../user.service';
import { catchError, tap } from 'rxjs/operators';
import { of } from 'rxjs';

@Component({
  selector: 'app-main',
  imports:[CommonModule],
  standalone: true,
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent implements OnInit {
  username: string | null=null;
  userID: number | null = null; // Assuming userID is stored in local storage
  history: Eval[] = []; // To store history data

  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  selectedFile: File | null = null;


  buttonStates = {
    fileUpload: false,
    uploadAndExtract: false,
    evaluateFile: false,
    restart: false,
  };
  disableButton(button: keyof typeof this.buttonStates) {
    this.buttonStates[button] = true;
  }

  constructor(private router: Router, private userService: UserService) {}

  ngOnInit(): void{
    const user = localStorage.getItem('userLoggedIn'); // Adjust key as per your localStorage structure
    if (user) {
      const userObj = JSON.parse(user);
      this.username = userObj.username; // Assuming the stored object has a `username` field
      this.userID = userObj.id; // Assuming user object has an `id` field
      this.loadHistory();
    }
  }

  activeOption: number = 1; // Default selected option
  isDropdownVisible: boolean = false;


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
        this.selectedFile = file;
        this.disableButton('fileUpload')
      } else {
        this.errorMessage = 'File must be a PDF.'; // Set error message
        this.selectedFileName = null; // Clear file name
      }
    }
  }

  
  

  triggerFileUpload(fileInput: HTMLInputElement): void {
    fileInput.click(); // Programmatically trigger the file input click
  }

  uploadExtractMsg: string | null = null; 
  uploadExtractComplete: boolean = false;


  uploadAndExtract():void{
    this.uploadExtractComplete = false
    this.uploadExtractMsg = 'Uploading...'
    setTimeout(() => { // Sleep the thread for 2 seconds
      this.uploadFile();
      this.uploadExtractMsg = 'Upload completed, Extracting text...';
      setTimeout(() => {
        this.extractFile();
        this.uploadExtractMsg = 'Text Extracted!';
        this.uploadExtractComplete = true
      }, 2000)
    }, 2000);
  }

  uploadFile():void{
    if (this.selectedFile) {
      this.userService
        .uploadFile(this.selectedFile)
        .pipe(
          // Log the response
          tap(response => {
            console.log('File upload response:', response);
          }),
          // Handle errors
          catchError(error => {
            console.error('Error during file upload:', error);
            alert('Failed to upload the file. Please try again.');
            return of(null); // Return an empty observable to gracefully handle the error
          })
        )
        .subscribe();
    } else {
      alert('No file selected. Please choose a file first.');
    }
  }

  extractFile():void{

  }

  evalMessage: string | null = null; 
  evalCompleted: boolean = false;

  evaluateFile(): void {
    this.evalMessage = "Evaluating..."
    this.evalCompleted = false
    setTimeout(() => {
      this.evalMessage = "Evaluation Completed!"
      this.evalCompleted = true
    }, 2000)

  }

  restart():void{
    window.location.reload();
    setTimeout(() => {
      this.buttonStates = {
        fileUpload: false,
        uploadAndExtract: false,
        evaluateFile: false,
        restart: false,
      };
    }, 500)
    
  }
  

  loadHistory(): void {
    if (this.userID !== null) {
      this.userService.getHistory(this.userID).pipe(
        tap((data: Eval[]) => {
          this.history = data
        }),
        catchError((error) => {
          console.error('Error fetching history:', error);
          return of(null); // Return a fallback value if needed
        })
      ).subscribe();
    }
  }

  viewReport(fileName: string): void {
    // Placeholder for "Report" button functionality
    console.log(`Viewing report for file: ${fileName}`);
    window.open('', '_blank');
  }
}
