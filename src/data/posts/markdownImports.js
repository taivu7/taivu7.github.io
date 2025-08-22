// Load markdown content from source files instead of hardcoded strings
export const loadMarkdownFromFiles = async (fileName) => {
  try {
    // Handle custom domain and GitHub Pages deployment
    let baseUrl = '';
    
    // In development, use localhost
    if (process.env.NODE_ENV === 'development') {
      baseUrl = '';
    } else {
      // In production, always use relative path for custom domains and GitHub Pages
      // Since we're deploying to a custom domain (taivu.dev), use relative path
      baseUrl = '';
    }
    
    const url = `${baseUrl}/posts/${fileName}`;
    console.log('Loading markdown from:', url, 'PUBLIC_URL:', process.env.PUBLIC_URL);
    
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