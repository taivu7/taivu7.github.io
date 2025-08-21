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
          <Link to="/">
            <div className="home-icon">
              <div className="home-roof"></div>
              <div className="home-base"></div>
            </div>
          </Link>
        </div>
        <ul className="nav-menu">
          <li className="nav-item">
            <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>Home</Link>
          </li>
          <li className="nav-item">
            <Link to="/blog" className={`nav-link ${location.pathname.startsWith('/blog') ? 'active' : ''}`}>Blog</Link>
          </li>
          <li className="nav-item">
            <Link to="/#contact" className="nav-link">Contact</Link>
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
            <p>&copy; 2025 Tai Vu. All rights reserved.</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
