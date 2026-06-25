import React, { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { getAllPosts } from '../data/posts';
import './Blog.css';

function formatMonthYear(date) {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    year: 'numeric',
  });
}

function Blog() {
  const [posts, setPosts] = useState([]);
  const [activeTag, setActiveTag] = useState('All');

  useEffect(() => {
    let active = true;
    getAllPosts()
      .then((all) => {
        if (active) setPosts(all);
      })
      .catch((err) => console.error('Error loading posts:', err));
    return () => {
      active = false;
    };
  }, []);

  // category pills: "All" + every unique tag across posts
  const categories = useMemo(() => {
    const tags = [...new Set(posts.flatMap((p) => p.tags || []))];
    return ['All', ...tags];
  }, [posts]);

  const filtered = useMemo(() => {
    if (activeTag === 'All') return posts;
    return posts.filter((p) => (p.tags || []).includes(activeTag));
  }, [posts, activeTag]);

  const featured = filtered[0];
  const rest = filtered.slice(1);

  return (
    <div className="writing">
      <div className="writing-inner">
        {/* header */}
        <header className="writing-header">
          <div className="writing-eyebrow">Writing</div>
          <h1 className="writing-title">Notes from production.</h1>
          <p className="writing-lead">
            Notes on machine learning, LLMs, and the messy reality of taking
            models from notebook to production.
          </p>
        </header>

        {/* category filters */}
        {categories.length > 1 && (
          <div className="writing-filters">
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                className={`writing-filter ${
                  activeTag === cat ? 'is-active' : ''
                }`}
                onClick={() => setActiveTag(cat)}
              >
                {cat}
              </button>
            ))}
          </div>
        )}

        {/* featured (most recent in current filter) */}
        {featured && (
          <Link to={`/blog/${featured.slug}`} className="writing-featured">
            <div className="writing-featured-meta">
              <span className="writing-featured-flag">FEATURED</span>
              <span className="writing-featured-sub">
                {[
                  featured.tags && featured.tags[0],
                  formatMonthYear(featured.date),
                  featured.readTime,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </span>
            </div>
            <h2 className="writing-featured-title">{featured.title}</h2>
            <p className="writing-featured-excerpt">{featured.excerpt}</p>
          </Link>
        )}

        {/* list */}
        <div className="writing-list">
          {rest.map((post) => (
            <Link
              key={post.slug}
              to={`/blog/${post.slug}`}
              className="writing-row"
            >
              <span className="writing-row-date">
                {formatMonthYear(post.date)}
              </span>
              <div>
                <div className="writing-row-title">{post.title}</div>
                <div className="writing-row-desc">{post.excerpt}</div>
              </div>
              <span className="writing-row-tag">
                {post.tags && post.tags[0]}
              </span>
            </Link>
          ))}
          {!featured && (
            <div className="writing-empty">No posts in this category yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Blog;
