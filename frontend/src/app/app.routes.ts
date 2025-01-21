import { provideRouter, Routes } from '@angular/router';
import { MainComponent } from './main/main.component';
import { LoginComponent } from './login/login.component';
import { SignupComponent } from './signup/signup.component';


export const routes: Routes = [
    {path: '', component:MainComponent},
    {path: 'login', component:LoginComponent},
    {path: 'signup', component: SignupComponent}
];

