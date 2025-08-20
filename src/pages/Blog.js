import React from 'react';
import { Link } from 'react-router-dom';
import BlogList from '../components/Blog/BlogList';

function Blog() {
  return (
    <div className="blog-page">
      <section className="blog-hero-section">
        <div className="container">
          <div className="blog-hero-content">
            <h1 className="blog-hero-title">Blog</h1>
            <p className="blog-hero-description">
              Thoughts on software development, technology, and life.
            </p>
            <Link to="/" className="btn btn-secondary">← Back to Home</Link>
          </div>
        </div>
      </section>
      
      <section className="blog-section">
        <div className="container">
          <BlogList />
        </div>
      </section>
    </div>
  );
}

export default Blog;