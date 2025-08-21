// Load markdown content from source files instead of hardcoded strings
export const loadMarkdownFromFiles = async (fileName) => {
  try {
    // In development, we can read from the actual markdown files
    // This requires the markdown files to be available in public folder for production
    const publicUrl = process.env.PUBLIC_URL || '';
    const url = `${publicUrl}/posts/${fileName}`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${fileName}: ${response.status}`);
    }
    
    return await response.text();
  } catch (error) {
    console.error('Error loading markdown file:', error);
    throw error;
  }
};