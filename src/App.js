import './App.css';

function App() {
  return (
    <div className="App">
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <div className="home-icon">
              <div className="home-roof"></div>
              <div className="home-base"></div>
            </div>
          </div>
          <ul className="nav-menu">
            <li className="nav-item">
              <a href="#home" className="nav-link">Home</a>
            </li>
            <li className="nav-item">
              <a href="#blog" className="nav-link">Blog</a>
            </li>
            <li className="nav-item">
              <a href="#contact" className="nav-link">Contact</a>
            </li>
          </ul>
        </div>
      </nav>

      <main className="main-content">
        <section id="home" className="hero-section">
          <div className="hero-container">
            <div className="hero-content">
              <h1 className="hero-title"><span className="wave-emoji">👋</span> I'm Tai</h1>
              <p className="hero-subtitle">
                Curious Engineer
              </p>
              <p className="hero-description">
                Welcome to my space on the web. I'm passionate about exploring new technologies, sharing my knowledge, experience, and exposure.
              </p>
              <div className="hero-buttons">
                <a href="#blog" className="btn btn-primary">View My Blog</a>
                <a href="#contact" className="btn btn-secondary">Get In Touch</a>
              </div>
            </div>
          </div>
        </section>

      </main>

      <footer className="footer">
        <div className="container">
          <p>&copy; 2025 Tai Vu. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
