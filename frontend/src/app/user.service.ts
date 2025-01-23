import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { catchError,map, Observable, of } from 'rxjs';
import { User } from './models/user';
import { Eval } from './models/eval';

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

  getHistory(UserIDFF: number) {

    const params = new HttpParams().set('UserID', UserIDFF.toString());

    return this.http.get<Eval[]>('http://127.0.0.1:5000/history', { params });
  }

  uploadFile(fileff: File) {
    // Create a FormData object to hold the file
    const formData = new FormData();
    formData.append('file', fileff);
  
    // Make an HTTP POST request to upload the file
    return this.http.post<{ message: string }>('http://127.0.0.1:5000/upload', formData);
  }

  extractFile() {
    return this.http.post<{ message: string }>('http://127.0.0.1:5000/extract', {});
  }
  evaluateFile() {
    return this.http.post<{ message: string }>('http://127.0.0.1:5000/evaluate', {});
  }
  
}
