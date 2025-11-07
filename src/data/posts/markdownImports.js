// Load markdown content from source files
export const loadMarkdownFromFiles = async (fileName) => {
  try {
    // For custom domains like taivu.dev, use relative path
    // PUBLIC_URL is empty for custom domains, so always use relative path
    const url = `/posts/${fileName}`;
    
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

// List of markdown files to import
export const markdownFiles = [
  '2025-11-03-LLM-Post-Training-Part-1.md',
  '2025-08-20-attention-mechanism-in-llm.md',
];

// Function to import all markdown files
export const importAllMarkdown = async () => {
  const importedFiles = {};
  for (const file of markdownFiles) {
    try {
      const content = await loadMarkdownFromFiles(file);
      importedFiles[file] = content;
    } catch (error) {
      console.error(`Error importing ${file}:`, error);
      importedFiles[file] = null;
    }
  }
  return importedFiles;
};

// Function to get all markdown files
export const getAllMarkdownFiles = async () => {
  return await importAllMarkdown();
};
