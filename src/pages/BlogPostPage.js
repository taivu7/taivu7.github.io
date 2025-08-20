import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPostBySlug } from '../data/posts';
import BlogPost from '../components/Blog/BlogPost';

function BlogPostPage() {
  const { slug } = useParams();
  const [post, setPost] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadPost = async () => {
      try {
        console.log('BlogPostPage - loading slug:', slug);
        setLoading(true);
        const foundPost = await getPostBySlug(slug);
        console.log('BlogPostPage - found post:', foundPost);
        setPost(foundPost);
        setError(null);
      } catch (err) {
        console.error('Error loading post:', err);
        setError(err.message);
        setPost(null);
      } finally {
        setLoading(false);
      }
    };

    if (slug) {
      loadPost();
    }
  }, [slug]);

  if (loading) {
    return (
      <div className="blog-post-page">
        <section className="blog-hero-section">
          <div className="container">
            <div className="blog-hero-content">
              <h1 className="blog-hero-title">Loading...</h1>
              <p className="blog-hero-description">Loading blog post...</p>
            </div>
          </div>
        </section>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="blog-post-page">
        <section className="blog-hero-section">
          <div className="container">
            <div className="blog-hero-content">
              <h1 className="blog-hero-title">Post Not Found</h1>
              <p className="blog-hero-description">
                {error || "The blog post you're looking for doesn't exist."}
              </p>
              <div className="blog-hero-buttons">
                <Link to="/blog" className="btn btn-primary">← Back to Blog</Link>
                <Link to="/" className="btn btn-secondary">Home</Link>
              </div>
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="blog-post-page">
      <section className="blog-post-nav">
        <div className="container">
          <Link to="/blog" className="back-to-blog">← Back to Blog</Link>
        </div>
      </section>
      
      <section className="blog-post-section">
        <div className="container">
          <BlogPost post={post} />
        </div>
      </section>
    </div>
  );
}

export default BlogPostPage;