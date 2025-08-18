# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a React application created with Create React App, intended for deployment to GitHub Pages (taivu7.github.io). The project includes gh-pages for deployment automation.

## Development Commands

### Core Development
- `npm start` - Start development server on http://localhost:3000 with hot reload
- `npm test` - Run tests in interactive watch mode using Jest and React Testing Library
- `npm run build` - Create production build in `build/` folder

### Testing
The project uses React Testing Library (@testing-library/react) with Jest DOM matchers. Test files follow the pattern `*.test.js`.

## Architecture

### Project Structure
- `/src` - React application source code
  - `App.js` - Main application component
  - `index.js` - React application entry point
  - `*.css` - Component and global styles
- `/public` - Static assets and HTML template
  - `index.html` - Main HTML template
  - `manifest.json` - PWA manifest
- `/build` - Production build output (generated)

### Key Dependencies
- React 19.1.1 with React DOM
- React Scripts 5.0.1 (Create React App toolchain)
- Testing: Jest, React Testing Library, Jest DOM
- Deployment: gh-pages package

## Deployment

The project includes gh-pages for automated deployment to GitHub Pages. The site is hosted at taivu7.github.io.

## Development Notes

- Standard Create React App configuration with React 19
- ESLint configured with react-app preset
- Modern browser targets in browserslist
- PWA-ready with manifest and service worker support through Create React App