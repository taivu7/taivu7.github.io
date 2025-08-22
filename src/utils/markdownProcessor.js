import { marked } from 'marked';
import katex from 'katex';
// Load markdown content directly from source files
import { loadMarkdownFromFiles } from '../data/posts/markdownImports';

// Configure marked options for better rendering
marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: true,
  mangle: false,
});

// Custom renderer for math expressions
const renderer = new marked.Renderer();

// Handle math blocks ($$...$$)
renderer.paragraph = function(text) {
  const mathBlockRegex = /^\$\$([\s\S]*?)\$\$$/;
  const match = text.match(mathBlockRegex);
  
  if (match) {
    try {
      const latex = match[1].trim();
      return `<div class="math-display">${katex.renderToString(latex, { displayMode: true })}</div>`;
    } catch (error) {
      console.error('KaTeX rendering error:', error);
      return `<div class="math-error">Math rendering error: ${error.message}</div>`;
    }
  }
  
  // Handle inline math ($...$)
  const inlineMathRegex = /\$([^$\n]+?)\$/g;
  const processedText = text.replace(inlineMathRegex, (_, latex) => {
    try {
      return katex.renderToString(latex, { displayMode: false });
    } catch (error) {
      console.error('KaTeX inline rendering error:', error);
      return `<span class="math-error">[Math Error: ${error.message}]</span>`;
    }
  });
  
  return `<p>${processedText}</p>`;
};

marked.use({ renderer });

// Simple frontmatter parser that works in the browser
function parseFrontmatter(content) {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);
  
  if (!match) {
    return {
      data: {},
      content: content
    };
  }
  
  const frontmatterText = match[1];
  const markdownContent = match[2];
  
  // Parse YAML-like frontmatter manually
  const frontmatter = {};
  const lines = frontmatterText.split('\n');
  
  for (const line of lines) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;
    
    const colonIndex = trimmedLine.indexOf(':');
    if (colonIndex === -1) continue;
    
    const key = trimmedLine.substring(0, colonIndex).trim();
    let value = trimmedLine.substring(colonIndex + 1).trim();
    
    // Remove quotes if present
    if ((value.startsWith('"') && value.endsWith('"')) || 
        (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    
    // Parse arrays (simple format: ["item1", "item2"])
    if (value.startsWith('[') && value.endsWith(']')) {
      try {
        value = JSON.parse(value);
      } catch (e) {
        // If JSON parsing fails, keep as string
      }
    }
    
    frontmatter[key] = value;
  }
  
  return {
    data: frontmatter,
    content: markdownContent
  };
}

export async function loadMarkdownPost(fileName) {
  console.log('loadMarkdownPost - fileName:', fileName);
  
  try {
    // Load markdown content directly from files
    const markdownContent = await loadMarkdownFromFiles(fileName);
    console.log('loadMarkdownPost - loaded from file, length:', markdownContent.length);
    
    // Parse frontmatter and content
    const { data: frontmatter, content } = parseFrontmatter(markdownContent);
    console.log('loadMarkdownPost - frontmatter:', frontmatter);
    
    // Convert markdown to HTML
    const htmlContent = marked(content);
    
    const result = {
      frontmatter,
      content: htmlContent,
      rawContent: content
    };
    
    console.log('loadMarkdownPost - final result:', result);
    return result;
  } catch (error) {
    console.error('Error in loadMarkdownPost:', error);
    return {
      frontmatter: {},
      content: '<p>Error loading content: ' + error.message + '</p>',
      rawContent: 'Error loading content'
    };
  }
}

// Function to dynamically load all posts from markdown files
export async function loadAllPosts() {
  try {
    const posts = [];
    
    // List of markdown files to load - updated to match actual files
    const postFiles = [
      '2025-08-20-attention-mechanism-in-llm.md'
    ];
    
    console.log('loadAllPosts - loading from markdown files:', postFiles);
    
    for (const fileName of postFiles) {
      try {
        const content = await loadMarkdownFromFiles(fileName);
        const { data: frontmatter } = parseFrontmatter(content);
        
        // Create post object with metadata
        const post = {
          id: posts.length + 1,
          title: frontmatter.title || 'Untitled',
          date: frontmatter.date || new Date().toISOString().split('T')[0],
          author: frontmatter.author || 'Unknown Author',
          excerpt: frontmatter.excerpt || '',
          tags: frontmatter.tags || [],
          readTime: frontmatter.readTime || '5 min read',
          featured: frontmatter.featured || false,
          slug: frontmatter.slug || fileName.replace('.md', ''),
          fileName: fileName,
          publishedDate: new Date(frontmatter.date || Date.now()),
        };
        
        posts.push(post);
        console.log('loadAllPosts - loaded post from file:', post.title);
      } catch (error) {
        console.error(`Error loading post ${fileName}:`, error);
      }
    }
    
    console.log('loadAllPosts - total posts loaded:', posts.length);
    return posts.sort((a, b) => b.publishedDate - a.publishedDate);
  } catch (error) {
    console.error('Error loading all posts:', error);
    return [];
  }
}

