import { useState, useEffect, useRef } from 'react';
import { loadMarkdownPost } from '../../utils/markdownProcessor';

function BlogPost({ post }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const contentRef = useRef(null);

  // Add copy buttons to code blocks
  useEffect(() => {
    if (!content || !contentRef.current) return;

    const addCopyButtons = () => {
      const codeBlocks = contentRef.current.querySelectorAll('pre');
      
      codeBlocks.forEach((block, index) => {
        // Skip if already has a copy button
        if (block.querySelector('.copy-button')) return;
        
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy code to clipboard');
        
        button.addEventListener('click', async () => {
          const code = block.querySelector('code');
          if (!code) return;
          
          try {
            await navigator.clipboard.writeText(code.textContent);
            button.textContent = 'Copied!';
            button.classList.add('copied');
            
            setTimeout(() => {
              button.textContent = 'Copy';
              button.classList.remove('copied');
            }, 2000);
          } catch (err) {
            console.error('Failed to copy code:', err);
            button.textContent = 'Failed';
            setTimeout(() => {
              button.textContent = 'Copy';
            }, 2000);
          }
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
      });
    };

    // Add copy buttons after a short delay to ensure DOM is ready
    const timer = setTimeout(addCopyButtons, 100);
    return () => clearTimeout(timer);
  }, [content]);

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
        ref={contentRef}
        className="blog-post-body"
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </article>
  );
}

export default BlogPost;