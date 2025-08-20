import React from 'react';
import { Link } from 'react-router-dom';

function BlogCard({ post }) {
  return (
    <Link to={`/blog/${post.slug}`} className="blog-card-link">
      <article className="blog-card">
      <div className="blog-card-content">
        <div className="blog-card-header">
          <h3 className="blog-card-title">{post.title}</h3>
          <div className="blog-card-meta">
            <time dateTime={post.date}>{new Date(post.date).toLocaleDateString()}</time>
            <span className="blog-card-read-time">{post.readTime}</span>
          </div>
        </div>
        <p className="blog-card-excerpt">{post.excerpt}</p>
        <div className="blog-card-tags">
          {post.tags.map(tag => (
            <span key={tag} className="blog-tag">{tag}</span>
          ))}
        </div>
      </div>
      </article>
    </Link>
  );
}

export default BlogCard;