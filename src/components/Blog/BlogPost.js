import { useState, useEffect } from 'react';
import { loadMarkdownPost } from '../../utils/markdownProcessor';

function BlogPost({ post }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadContent = async () => {
      try {
        console.log('Loading post:', post.fileName); // Debug log
        const markdownData = await loadMarkdownPost(post.fileName);
        console.log('Loaded markdown data:', markdownData); // Debug log
        setContent(markdownData.content);
      } catch (error) {
        console.error('Error loading post content:', error);
        setContent('<p>Error loading post content.</p>');
      } finally {
        setLoading(false);
      }
    };

    if (post && post.fileName) {
      loadContent();
    } else {
      console.error('No post or fileName provided:', post);
      setContent('<p>No post data available.</p>');
      setLoading(false);
    }
  }, [post, post.fileName]);

  if (loading) {
    return <div className="blog-post-loading">Loading...</div>;
  }

  return (
    <article id={`post-${post.slug}`} className="blog-post">
      <header className="blog-post-header">
        <h1 className="blog-post-title">{post.title}</h1>
        <div className="blog-post-meta">
          <div className="blog-post-author">By {post.author}</div>
          <time className="blog-post-date" dateTime={post.date}>
            {new Date(post.date).toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </time>
          <span className="blog-post-read-time">{post.readTime}</span>
        </div>
        <div className="blog-post-tags">
          {post.tags.map(tag => (
            <span key={tag} className="blog-tag">{tag}</span>
          ))}
        </div>
      </header>
      <div 
        className="blog-post-body"
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </article>
  );
}

export default BlogPost;