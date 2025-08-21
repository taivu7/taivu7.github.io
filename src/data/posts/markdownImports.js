// Load markdown content from source files instead of hardcoded strings
export const loadMarkdownFromFiles = async (fileName) => {
  try {
    // Handle custom domain and GitHub Pages deployment
    // For custom domains, we need to use the full domain path
    let baseUrl = '';
    
    // In development, use localhost
    if (process.env.NODE_ENV === 'development') {
      baseUrl = '';
    } else {
      // In production, use relative path that works with custom domains
      baseUrl = process.env.PUBLIC_URL || '';
    }
    
    const url = `${baseUrl}/posts/${fileName}`;
    console.log('Loading markdown from:', url);
    
    const response = await fetch(url);
    if (!response.ok) {
      console.error(`Failed to fetch ${fileName}: ${response.status} ${response.statusText}`);
      throw new Error(`Failed to fetch ${fileName}: ${response.status}`);
    }
    
    const content = await response.text();
    console.log(`Successfully loaded ${fileName}, content length:`, content.length);
    return content;
  } catch (error) {
    console.error('Error loading markdown file:', error);
    throw error;
  }
};