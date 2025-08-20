import { marked } from 'marked';

// Configure marked options for better rendering
marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: true,
  mangle: false,
});

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
    // Load content from public directory via fetch
    const response = await fetch(`/posts/${fileName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${fileName}: ${response.status}`);
    }
    
    const markdownContent = await response.text();
    console.log('loadMarkdownPost - loaded content, length:', markdownContent.length);
    
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

// Function to dynamically load all posts from public directory
export async function loadAllPosts() {
  try {
    // For now, we'll need to manually maintain a list of post files
    // GitHub Pages doesn't support directory listing
    const postFiles = [
      '2025-08-20-attention-mechanism-in-llm.md'
    ];
    
    const posts = [];
    
    for (const fileName of postFiles) {
      try {
        const response = await fetch(`/posts/${fileName}`);
        if (!response.ok) {
          console.warn(`Failed to load ${fileName}: ${response.status}`);
          continue;
        }
        
        const content = await response.text();
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
      } catch (error) {
        console.error(`Error loading post ${fileName}:`, error);
      }
    }
    
    return posts.sort((a, b) => b.publishedDate - a.publishedDate);
  } catch (error) {
    console.error('Error loading all posts:', error);
    return [];
  }
}

