// Blog posts metadata and exports
// This file serves as the central registry for all blog posts
import { loadAllPosts } from '../../utils/markdownProcessor';

let postsCache = null;
let postsPromise = null;

// Helper function to load posts with caching
async function loadPosts() {
  if (postsCache) {
    return postsCache;
  }
  
  if (!postsPromise) {
    postsPromise = loadAllPosts();
  }
  
  postsCache = await postsPromise;
  return postsCache;
}

// Helper functions for blog post management
export const getAllPosts = async () => {
  const posts = await loadPosts();
  return posts.sort((a, b) => b.publishedDate - a.publishedDate);
};

export const getFeaturedPosts = async () => {
  const posts = await loadPosts();
  return posts.filter(post => post.featured);
};

export const getPostBySlug = async (slug) => {
  console.log('getPostBySlug - searching for slug:', slug);
  const posts = await loadPosts();
  console.log('getPostBySlug - available posts:', posts.map(p => p.slug));
  const found = posts.find(post => post.slug === slug);
  console.log('getPostBySlug - found:', found);
  return found;
};

export const getPostsByTag = async (tag) => {
  const posts = await loadPosts();
  return posts.filter(post => post.tags.includes(tag));
};

export const getAllTags = async () => {
  const posts = await loadPosts();
  const allTags = posts.flatMap(post => post.tags);
  return [...new Set(allTags)].sort();
};

export const getRecentPosts = async (limit = 5) => {
  const posts = await getAllPosts();
  return posts.slice(0, limit);
};

// Synchronous version for backwards compatibility (will be deprecated)
export const getPostBySlugSync = (slug) => {
  console.warn('getPostBySlugSync is deprecated, use getPostBySlug instead');
  // Return null for now, components should use async version
  return null;
};