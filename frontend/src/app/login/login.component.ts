import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { UserService } from '../user.service';
import { User } from '../models/user';
import { catchError, tap } from 'rxjs/operators';
import { of } from 'rxjs';


@Component({
  selector: 'app-login',
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.css',
  standalone:true
})
export class LoginComponent {
  email: string = '';
  password: string = '';
  message: string = ''

  constructor(private router:Router, private userService: UserService){}

  ngOnInit(): void {
    localStorage.clear()
    sessionStorage.clear()
  }

  onLogin(): void {

    this.userService.login(this.email, this.password).pipe(
      tap((userFromDB: User) => {
        if (userFromDB) {
          localStorage.setItem('userLoggedIn', JSON.stringify(userFromDB));
          this.router.navigate(['']);
        } else {
          this.message = "Error, incorrect email or password";
        }
      }),
      catchError((error) => {
        this.message = "Error, incorrect email or password"; // Handle the error
        console.error('Login failed:', error);
        return of(null); // Return a fallback value if needed
      })
    ).subscribe();

  }
  back() : void{
    this.router.navigate([''])
  }
}
