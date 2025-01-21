import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';
import { MainComponent } from './app/main/main.component';
import { provideHttpClient } from '@angular/common/http';
import { provideRouter } from '@angular/router';
import { routes } from './app/app.routes';

bootstrapApplication(MainComponent);

bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes), // Provide routing
    provideHttpClient(),   // Provide HttpClient for API calls
  ],
}).catch((err) => console.error(err));
