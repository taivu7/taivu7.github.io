import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import './App.css';
import Home from './pages/Home';
import Blog from './pages/Blog';
import BlogPostPage from './pages/BlogPostPage';

function Navbar() {
  const location = useLocation();
  
  return (
    <nav className="navbar">
      <div className="nav-container">
        <div className="nav-logo">
          <Link to="/" className="nav-wordmark">Tai Vu</Link>
        </div>
        <ul className="nav-menu">
          <li className="nav-item">
            <Link to="/blog" className={`nav-link ${location.pathname.startsWith('/blog') ? 'active' : ''}`}>Writing</Link>
          </li>
          <li className="nav-item">
            <Link to="/#experience" className="nav-link">Experience</Link>
          </li>
          <li className="nav-item">
            <Link to="/#about" className="nav-link">About</Link>
          </li>
          <li className="nav-item">
            <Link to="/#contact" className="nav-link nav-link-accent">Contact</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

function ScrollToHash() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash) {
      // Small delay to ensure the element is rendered
      setTimeout(() => {
        const element = document.getElementById(location.hash.substring(1));
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    }
  }, [location]);

  return null;
}

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <ScrollToHash />
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/blog" element={<Blog />} />
            <Route path="/blog/:slug" element={<BlogPostPage />} />
          </Routes>
        </main>

        <footer className="footer">
          <div className="container">
            <p>Tai Vu — Ho Chi Minh City</p>
            <span className="footer-meta">© 2026</span>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
