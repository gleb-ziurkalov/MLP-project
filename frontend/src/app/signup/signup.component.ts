import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { User } from '../models/user';
import { UserService } from '../user.service';
import { catchError, tap } from 'rxjs/operators';
import { of } from 'rxjs';

@Component({
  selector: 'app-signup',
  imports: [CommonModule, FormsModule],
  templateUrl: './signup.component.html',
  styleUrl: './signup.component.css',
  standalone: true
})
export class SignupComponent {
  email: string = '';
  username: string = '';
  password: string = '';
  confirmPassword: string = '';
  message: string = ''

  constructor(private router:Router,private userService:UserService){}

  onSignup(): void {
    if (this.password !== this.confirmPassword) {
      this.message = 'Passwords do not match!'
      return;
    }
    if (this.password == '' || this.confirmPassword=='' || this.email=='' || this.username==''){
      this.message = 'All fields must be filled!'
      return;
    }


    const rgxmail: RegExp = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

    if (!rgxmail.test(this.email)) {
      this.message = 'Incorrect email form';
      return;
    }


    this.userService.signup(this.email, this.username, this.password).pipe(
      tap(() => {
          this.router.navigate(['']);
       
      }),
      catchError((error) => {
        this.message = "Error, username or email aready exist!"; // Handle the error
        console.error('Signup failed:', error);
        return of(null); // Return a fallback value if needed
      })
    ).subscribe();
  }
  back() : void{
    this.router.navigate([''])
  }
}
