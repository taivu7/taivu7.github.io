import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPostBySlug, getAllPosts } from '../data/posts';
import { loadMarkdownPost } from '../utils/markdownProcessor';
import './Article.css';

const AUTHOR_BIO =
  'Senior Data Scientist with 9+ years across telecom, cybersecurity, and SaaS — building machine-learning systems and data pipelines that hold up in production.';

function formatLongDate(date) {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function BlogPostPage() {
  const { slug } = useParams();
  const [post, setPost] = useState(null);
  const [content, setContent] = useState('');
  const [others, setOthers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const contentRef = useRef(null);

  // Load post, its markdown body, and sibling posts
  useEffect(() => {
    let active = true;
    if (!slug) return;

    const load = async () => {
      try {
        setLoading(true);
        setContent('');
        const found = await getPostBySlug(slug);
        if (!active) return;

        if (!found) {
          setError('not-found');
          setPost(null);
          return;
        }

        setPost(found);
        setError(null);

        const [markdown, all] = await Promise.all([
          loadMarkdownPost(found.fileName),
          getAllPosts(),
        ]);
        if (!active) return;

        setContent(markdown.content);
        setOthers(all.filter((p) => p.slug !== found.slug).slice(0, 2));
      } catch (err) {
        if (active) {
          console.error('Error loading post:', err);
          setError(err.message || 'error');
        }
      } finally {
        if (active) setLoading(false);
      }
    };

    load();
    window.scrollTo(0, 0);
    return () => {
      active = false;
    };
  }, [slug]);

  // Attach copy buttons to code blocks once content is rendered
  useEffect(() => {
    if (!content || !contentRef.current) return;
    const root = contentRef.current;

    const timer = setTimeout(() => {
      root.querySelectorAll('pre').forEach((block) => {
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
            button.textContent = 'Failed';
            setTimeout(() => {
              button.textContent = 'Copy';
            }, 2000);
          }
        });
        block.appendChild(button);
      });
    }, 80);

    return () => clearTimeout(timer);
  }, [content]);

  if (loading) {
    return (
      <div className="article">
        <div className="article-status">
          <p>Loading…</p>
        </div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="article">
        <div className="article-status">
          <h1>Post not found</h1>
          <p>The article you're looking for doesn't exist.</p>
          <Link to="/blog" className="article-status-link">
            ← All writing
          </Link>
        </div>
      </div>
    );
  }

  const primaryTag = post.tags && post.tags[0];

  return (
    <div className="article">
      <article className="article-shell">
        {/* header */}
        <header className="article-header">
          <Link to="/blog" className="article-back">
            ← All writing
          </Link>
          <div className="article-meta">
            {primaryTag && (
              <>
                <span className="article-meta-tag">{primaryTag}</span>
                <span>·</span>
              </>
            )}
            <span>{formatLongDate(post.date)}</span>
            {post.readTime && (
              <>
                <span>·</span>
                <span>{post.readTime}</span>
              </>
            )}
          </div>
          <h1 className="article-title">{post.title}</h1>
          {post.excerpt && <p className="article-subtitle">{post.excerpt}</p>}
          <div className="article-byline">
            <div className="article-avatar">TV</div>
            <div className="article-byline-text">
              <span className="article-byline-name">
                {post.author || 'Tai Vu'}
              </span>
              <span className="article-byline-role">Senior Data Scientist</span>
            </div>
          </div>
        </header>

        {/* body */}
        <div
          ref={contentRef}
          className="article-prose"
          dangerouslySetInnerHTML={{ __html: content }}
        />

        {/* tags */}
        {post.tags && post.tags.length > 0 && (
          <div className="article-tags">
            {post.tags.map((tag) => (
              <span key={tag} className="article-tag">
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* author card */}
        <div className="article-author">
          <div className="article-author-avatar">TV</div>
          <div>
            <div className="article-author-name">
              Written by {post.author || 'Tai Vu'}
            </div>
            <p className="article-author-bio">{AUTHOR_BIO}</p>
            <div className="article-author-links">
              <a href="mailto:me@taivu.dev">Email</a>
              <a
                href="https://github.com/taivu7"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
              <a
                href="https://www.linkedin.com/in/tai-vu"
                target="_blank"
                rel="noopener noreferrer"
              >
                LinkedIn
              </a>
            </div>
          </div>
        </div>

        {/* keep reading */}
        {others.length > 0 && (
          <div className="article-next">
            <h2 className="article-next-label">Keep reading</h2>
            <div className="article-next-list">
              {others.map((p) => (
                <Link
                  key={p.slug}
                  to={`/blog/${p.slug}`}
                  className="article-next-row"
                >
                  <div>
                    <div className="article-next-title">{p.title}</div>
                    <div className="article-next-desc">{p.excerpt}</div>
                  </div>
                  <span className="article-next-tag">
                    {p.tags && p.tags[0]}
                  </span>
                </Link>
              ))}
            </div>
          </div>
        )}
      </article>
    </div>
  );
}

export default BlogPostPage;
