# Project: Personal Portfolio Website

## Project Overview

This project is a personal portfolio website for Tai Vu, a "Curious Engineer". It's a single-page application built with React. The website features a navigation bar, a hero section with a brief introduction, a contact section with links to email and social media, and a footer. The project was bootstrapped with Create React App.

## Building and Running

### Prerequisites

*   Node.js and npm (or yarn)

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    npm install
    ```

### Running the application

*   **Development:**
    ```bash
    npm start
    ```
    This will start the development server at `http://localhost:3000`.

*   **Production Build:**
    ```bash
    npm run build
    ```
    This will create a production-ready build in the `build` directory.

### Deployment

The project is set up for deployment to GitHub Pages.

```bash
npm run deploy
```

This command will first build the application and then deploy the `build` directory to the `gh-pages` branch of the GitHub repository. The website is configured to be available at `https://taivu.dev`.

## Development Conventions

*   **Code Style:** The project follows the standard React coding conventions. ESLint is configured to enforce the `react-app` and `react-app/jest` styles.
*   **Testing:** The project is set up with React Testing Library. Tests can be run with:
    ```bash
    npm test
    ```
