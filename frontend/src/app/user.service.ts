import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { catchError,map, of } from 'rxjs';
import { User } from './models/user';

@Injectable({
  providedIn: 'root'
})
export class UserService {

  constructor(private http: HttpClient) { }

  login(emailFF: string, passwordFF:string ) {

    const payload = { email: emailFF, password: passwordFF };

    return this.http.post<User>('http://127.0.0.1:5000/login', payload);
  }

  signup(emailFF: string, usernameFF:string, passwordFF:string ) {

    const payload = { email: emailFF, username: usernameFF, password: passwordFF };

    return this.http.post<User>('http://127.0.0.1:5000/signup', payload);
  }
}
