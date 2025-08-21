import React, { useState, useEffect } from 'react';
import BlogCard from './BlogCard';
import { getAllPosts } from '../../data/posts';

function BlogList() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadPosts = async () => {
      try {
        setLoading(true);
        const allPosts = await getAllPosts();
        setPosts(allPosts);
        setError(null);
      } catch (err) {
        console.error('Error loading posts:', err);
        setError(err.message);
        setPosts([]);
      } finally {
        setLoading(false);
      }
    };

    loadPosts();
  }, []);

  if (loading) {
    return (
      <div className="blog-list">
        <div className="blog-header">
          <h2 className="blog-title">Latest Posts</h2>
          <p className="blog-description">Loading posts...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="blog-list">
        <div className="blog-header">
          <h2 className="blog-title">Latest Posts</h2>
          <p className="blog-description">Error loading posts: {error}</p>
        </div>
      </div>
    );
  }

  if (posts.length === 0) {
    return (
      <div className="blog-list">
        <div className="blog-header">
          <h2 className="blog-title">Latest Posts</h2>
          <p className="blog-description">No posts found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="blog-list">
      <div className="blog-header">
        <h2 className="blog-title">Latest Posts</h2>
        <p className="blog-description">
          Thoughts on software development, technology, and life.
        </p>
      </div>
      <div className="blog-grid">
        {posts.map(post => (
          <BlogCard key={post.id} post={post} />
        ))}
      </div>
    </div>
  );
}

export default BlogList;