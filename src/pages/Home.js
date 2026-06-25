import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getRecentPosts } from '../data/posts';
import './Landing.css';

const EXPERIENCE = [
  {
    role: 'Data Team Lead · MobiFone',
    detail:
      'Leading ML models and centralized data pipelines that lift customer engagement at telecom scale.',
    period: '2020 — Now',
  },
  {
    role: 'Senior Data Scientist · Opusmatch',
    detail:
      'Advisor (US SaaS) — predictive and prescriptive models, data quality, and algorithm development.',
    period: '2023 — 2025',
  },
  {
    role: 'Senior Data Scientist · VinCSS',
    detail:
      'Applied ML and statistical methods to threat detection and product security.',
    period: '2019 — 2020',
  },
  {
    role: 'Data Scientist · Zalo',
    detail:
      'ML models and A/B testing for user-facing product features, with real-time monitoring pipelines.',
    period: '2016 — 2019',
  },
];

function formatMonthYear(date) {
  return new Date(date).toLocaleDateString('en-US', {
    month: 'short',
    year: 'numeric',
  });
}

function Home() {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    let active = true;
    getRecentPosts(3)
      .then((recent) => {
        if (active) setPosts(recent);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="landing">
      <div className="landing-inner">
        {/* hero */}
        <section id="about" className="landing-hero">
          <div className="landing-eyebrow">Senior Data Scientist</div>
          <h1 className="landing-hero-title landing-hero-quote">
            “But when he comes, the Spirit of truth, he will guide you to all
            truth. He will not speak on his own, but he will speak what he hears,
            and will declare to you the things that are coming.”
          </h1>
          <div className="landing-hero-cite">— John 16:13</div>
          <p className="landing-hero-lead">
            Nine years turning machine learning and statistics into decisions
            people trust — across telecom, cybersecurity, and SaaS. I lead data
            teams and ship models that hold up in production.
          </p>
        </section>

        {/* recent writing */}
        <section id="writing">
          <div className="landing-section-head">
            <h2 className="landing-section-label">Recent writing</h2>
            <Link to="/blog" className="landing-section-link">
              All articles →
            </Link>
          </div>
          <div className="landing-writing">
            {posts.map((post) => (
              <Link
                key={post.slug}
                to={`/blog/${post.slug}`}
                className="landing-article"
              >
                <span className="landing-article-date">
                  {formatMonthYear(post.date)}
                </span>
                <div>
                  <div className="landing-article-title">{post.title}</div>
                  <div className="landing-article-desc">{post.excerpt}</div>
                </div>
                <span className="landing-article-tag">
                  {post.tags && post.tags[0]}
                </span>
              </Link>
            ))}
          </div>
        </section>

        {/* experience */}
        <section id="experience" className="landing-experience">
          <h2 className="landing-section-label">Experience</h2>
          <div className="landing-exp-list">
            {EXPERIENCE.map((item) => (
              <div key={item.role} className="landing-exp-row">
                <div>
                  <div className="landing-exp-role">{item.role}</div>
                  <div className="landing-exp-detail">{item.detail}</div>
                </div>
                <span className="landing-exp-period">{item.period}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* contact */}
      <section id="contact" className="landing-contact">
        <div className="landing-contact-inner">
          <h2 className="landing-contact-title">
            Building something with data? Let's talk.
          </h2>
          <p className="landing-contact-lead">
            I'm always happy to trade notes on forecasting, NLP, or keeping
            models alive in production.
          </p>
          <div className="landing-contact-actions">
            <a
              href="mailto:me@taivu.dev"
              className="landing-btn landing-btn-primary"
            >
              Email me
            </a>
            <a
              href="https://github.com/taivu7"
              target="_blank"
              rel="noopener noreferrer"
              className="landing-btn landing-btn-ghost"
            >
              GitHub
            </a>
            <a
              href="https://www.linkedin.com/in/tai-vu"
              target="_blank"
              rel="noopener noreferrer"
              className="landing-btn landing-btn-ghost"
            >
              LinkedIn
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
