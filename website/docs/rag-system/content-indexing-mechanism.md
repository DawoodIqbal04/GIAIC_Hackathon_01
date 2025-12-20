# Content Indexing Mechanism During Build Process

## Overview
This document outlines the implementation of the content indexing mechanism that will run during the Docusaurus build process to automatically index all documentation content into the vector database.

## Implementation Strategy

The content indexing mechanism will be implemented as a Node.js script that runs during the Docusaurus build process. The script will:

1. Parse all markdown files in the documentation
2. Extract content and metadata
3. Chunk the content appropriately
4. Generate embeddings
5. Store in the vector database

## Content Parser Script

Create `scripts/index-content.js`:

```javascript
const fs = require('fs');
const path = require('path');
const glob = require('glob');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const weaviate = require('weaviate-client');

// Configuration
const DOCS_DIR = './docs';
const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 100;

// Initialize Weaviate client
const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

// Text splitter for chunking content
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

/**
 * Extract metadata from markdown file
 */
function extractMetadata(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Extract frontmatter if present
  const frontmatterRegex = /^---\s*\n(.*?)\n---\s*\n/s;
  const frontmatterMatch = content.match(frontmatterRegex);
  
  let metadata = {};
  if (frontmatterMatch) {
    const frontmatter = frontmatterMatch[1];
    // Simple frontmatter parser (in production, use a proper YAML parser)
    const lines = frontmatter.split('\n');
    lines.forEach(line => {
      const [key, value] = line.split(': ');
      if (key && value) {
        metadata[key.trim()] = value.trim().replace(/^["']|["']$/g, '');
      }
    });
  }
  
  // Extract module and chapter from path
  const pathParts = filePath.split(path.sep);
  const moduleIndex = pathParts.indexOf('docs') + 1;
  if (moduleIndex < pathParts.length) {
    metadata.module = pathParts[moduleIndex];
    
    if (moduleIndex + 1 < pathParts.length) {
      const chapterPart = pathParts[moduleIndex + 1];
      if (chapterPart.includes('chapter')) {
        metadata.chapter = chapterPart;
      }
    }
  }
  
  return metadata;
}

/**
 * Extract content from markdown file, excluding frontmatter
 */
function extractContent(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  
  // Remove frontmatter if present
  const frontmatterRegex = /^---\s*\n(.*?)\n---\s*\n/s;
  const contentWithoutFrontmatter = content.replace(frontmatterRegex, '');
  
  // Remove markdown syntax for better semantic search
  // This is a simplified version - in production, use a proper markdown parser
  return contentWithoutFrontmatter
    .replace(/#{1,6}\s+/g, '') // Remove headers
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
    .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
    .replace(/\*([^*]+)\*/g, '$1') // Remove italic
    .replace(/!\[[^\]]*\]\([^)]+\)/g, '') // Remove image tags
    .replace(/`([^`]+)`/g, '$1') // Remove inline code
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/\n{3,}/g, '\n\n') // Normalize multiple newlines
    .trim();
}

/**
 * Process a single markdown file
 */
async function processFile(filePath) {
  console.log(`Processing file: ${filePath}`);
  
  try {
    const metadata = extractMetadata(filePath);
    const content = extractContent(filePath);
    
    // Split content into chunks
    const chunks = await textSplitter.splitText(content);
    
    // Get the section title from the first H1 or H2 in the content
    const titleMatch = content.match(/^(#{1,2}\s+.*$)/m);
    const sectionTitle = titleMatch ? titleMatch[1].replace(/#{1,2}\s+/, '').trim() : 'Untitled';
    
    // Index each chunk
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      // Skip empty chunks
      if (!chunk.trim()) continue;
      
      // Create data object for Weaviate
      const dataObj = {
        content: chunk,
        sourceDocument: path.basename(filePath),
        sectionTitle: sectionTitle,
        module: metadata.module || 'Unknown Module',
        chapter: metadata.chapter || 'Unknown Chapter',
        chunkIndex: i
      };
      
      // Add to Weaviate
      await client.data
        .creator()
        .withClassName('DocChunk')
        .withProperties(dataObj)
        .do();
        
      console.log(`  Indexed chunk ${i+1}/${chunks.length} from ${path.basename(filePath)}`);
    }
    
    console.log(`Finished processing: ${filePath}`);
  } catch (error) {
    console.error(`Error processing file ${filePath}:`, error);
  }
}

/**
 * Index all markdown files in the docs directory
 */
async function indexAllDocs() {
  console.log('Starting content indexing...');
  
  // Find all markdown files in docs directory
  const pattern = path.join(DOCS_DIR, '**/*.md');
  const files = glob.sync(pattern, {
    ignore: ['**/node_modules/**', '**/build/**']
  });
  
  console.log(`Found ${files.length} markdown files to process`);
  
  // Process each file
  for (const file of files) {
    await processFile(file);
  }
  
  console.log('Content indexing completed!');
}

/**
 * Clear existing content from Weaviate before re-indexing
 */
async function clearExistingContent() {
  console.log('Clearing existing content from Weaviate...');
  
  try {
    await client.batch
      .objectsBatchDeleter()
      .withClassName('DocChunk')
      .do();
      
    console.log('Existing content cleared');
  } catch (error) {
    console.error('Error clearing existing content:', error);
  }
}

// Main execution
async function main() {
  // Clear existing content first
  await clearExistingContent();
  
  // Index all documentation
  await indexAllDocs();
}

// Run if this file is executed directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  extractMetadata,
  extractContent,
  processFile,
  indexAllDocs
};
```

## Package Dependencies

Add to `package.json`:

```json
{
  "scripts": {
    "index-content": "node scripts/index-content.js",
    "build": "npm run index-content && docusaurus build"
  },
  "dependencies": {
    "weaviate-client": "^2.1.0",
    "langchain": "^0.0.182",
    "glob": "^10.3.3"
  }
}
```

## Docusaurus Configuration Integration

Update `docusaurus.config.js` to include the indexing script in the build process:

```javascript
// docusaurus.config.js
const config = {
  // ... other config options

  scripts: [
    // ... other scripts
  ],
  
  plugins: [
    // ... other plugins
    
    // Add a plugin to run content indexing during build
    async function contentIndexingPlugin(context, options) {
      return {
        name: 'content-indexing-plugin',
        async postBuild(props) {
          console.log('Starting content indexing after build...');
          
          // Import and run the indexing function
          const { indexAllDocs } = require('./scripts/index-content');
          await indexAllDocs();
          
          console.log('Content indexing completed!');
        },
      };
    },
  ],
  
  // ... rest of config
};

module.exports = config;
```

## Alternative: Using a Build Hook

Instead of a plugin, you could also use a build hook by modifying the build script in `package.json`:

```json
{
  "scripts": {
    "prebuild": "node scripts/index-content.js",
    "build": "docusaurus build",
    "serve": "docusaurus serve"
  }
}
```

## Chunking Strategy

The content will be chunked using a recursive character text splitter with the following parameters:

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 100 characters
- **Separators**: ['\\n\\n', '\\n', ' ', '']

This strategy ensures that:
1. Chunks are of manageable size for embedding
2. Context is preserved through overlap
3. Natural breaks in content are respected

## Metadata Extraction

For each document chunk, the following metadata will be extracted:

1. **Source Document**: The filename of the original document
2. **Section Title**: The H1 or H2 title of the section
3. **Module**: The module the content belongs to (extracted from path)
4. **Chapter**: The specific chapter (extracted from path)
5. **Chunk Index**: The position of this chunk in the document

## Error Handling and Monitoring

The indexing process includes:

1. **Individual file error handling**: If one file fails, continue with others
2. **Progress logging**: Log each file and chunk as it's processed
3. **Duplicate handling**: Clear existing content before re-indexing
4. **Validation**: Verify content was properly indexed

## Performance Considerations

1. **Batch Processing**: Process files sequentially to avoid overwhelming the vector database
2. **Rate Limiting**: Add delays between requests if needed
3. **Memory Management**: Process large files in chunks to avoid memory issues
4. **Incremental Updates**: In production, implement incremental updates based on file modification times

## Testing the Indexing Process

Create a test script to verify the indexing works correctly:

```javascript
// test-indexing.js
const { indexAllDocs, clearExistingContent } = require('./scripts/index-content');
const weaviate = require('weaviate-client');

const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

async function testIndexing() {
  try {
    // Clear existing content
    await clearExistingContent();
    
    // Count documents before indexing
    const resultBefore = await client.graphql
      .aggregate()
      .withClassName('DocChunk')
      .withFields('meta { count }')
      .do();
      
    console.log(`Documents before indexing: ${resultBefore.data.Aggregate.DocChunk[0].meta.count}`);
    
    // Run indexing
    await indexAllDocs();
    
    // Count documents after indexing
    const resultAfter = await client.graphql
      .aggregate()
      .withClassName('DocChunk')
      .withFields('meta { count }')
      .do();
      
    console.log(`Documents after indexing: ${resultAfter.data.Aggregate.DocChunk[0].meta.count}`);
    
    // Test search
    const searchResult = await client.graphql
      .get()
      .withClassName('DocChunk')
      .withFields('content sourceDocument module chapter')
      .withLimit(3)
      .do();
      
    console.log('Sample indexed documents:');
    searchResult.data.Get.DocChunk.slice(0, 3).forEach((doc, i) => {
      console.log(`${i+1}. Module: ${doc.module}, Chapter: ${doc.chapter}`);
      console.log(`   Source: ${doc.sourceDocument}`);
      console.log(`   Content preview: ${doc.content.substring(0, 100)}...`);
      console.log('---');
    });
    
  } catch (error) {
    console.error('Error during indexing test:', error);
  }
}

if (require.main === module) {
  testIndexing();
}
```

This implementation provides a robust mechanism to automatically index all documentation content during the Docusaurus build process, ensuring that the RAG system has access to the most current content.