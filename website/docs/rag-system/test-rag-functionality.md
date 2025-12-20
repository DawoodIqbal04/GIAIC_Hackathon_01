# Testing RAG Functionality with Complete Book Content

## Overview
This document outlines the comprehensive testing approach for the RAG (Retrieval Augmented Generation) functionality with the complete Physical AI and Humanoid Robotics Technical Book content. The testing ensures that the system can effectively retrieve relevant information and generate accurate responses based on the entire book content.

## Test Strategy

### 1. Functional Testing
- Verify content indexing of all book modules and chapters
- Test semantic search accuracy across the entire content
- Validate conversation management and history
- Confirm proper source citation functionality

### 2. Performance Testing
- Measure response times for various query types
- Test system behavior under load
- Verify scalability with increasing content volume

### 3. Accuracy Testing
- Validate response accuracy against source content
- Test for hallucinations and factual correctness
- Ensure proper handling of ambiguous queries

## Test Scenarios

### Scenario 1: Content Indexing Verification
**Objective**: Verify all book content is properly indexed in the vector database

**Test Steps**:
1. Run the complete content indexing process
2. Verify all 4 modules and 16 chapters are indexed
3. Check that metadata (module, chapter, section) is correctly preserved
4. Validate that content chunks are properly segmented

**Expected Results**:
- All documentation files are indexed without errors
- Total document count matches expected number
- Metadata fields are correctly populated
- Content is accessible via semantic search

### Scenario 2: Basic Search Functionality
**Objective**: Test semantic search across the complete book content

**Test Queries**:
1. "What is ROS 2 middleware?"
2. "Explain Isaac Sim setup process"
3. "How does Nav2 work for humanoid robots?"
4. "What are the key differences between ROS 1 and ROS 2?"
5. "Explain perception networks in robotics"

**Expected Results**:
- Relevant content chunks returned within 2 seconds
- Top results directly address the query
- Source documents correctly identified
- Proper context provided for each result

### Scenario 3: Chat Conversation Flow
**Objective**: Test end-to-end chat functionality with book content

**Test Steps**:
1. Start a new conversation
2. Ask a series of related questions about a specific topic
3. Verify context is maintained across conversation
4. Test follow-up questions that reference previous responses

**Sample Conversation Flow**:
1. "What is the focus of Module 1?"
2. "Tell me about ROS 2 nodes"
3. "How do nodes communicate with each other?"
4. "What's the difference between topics and services?"

**Expected Results**:
- Each response is based on book content
- Context is maintained throughout conversation
- Source citations are provided for each response
- Follow-up questions are understood in context

### Scenario 4: Cross-Module Queries
**Objective**: Test the system's ability to retrieve information across different modules

**Test Queries**:
1. "How does the perception system from Module 3 integrate with the navigation in Module 4?"
2. "Compare the simulation approaches in Module 2 with the real-world deployment in Module 1"
3. "How do the VLA systems in Module 4 build on the ROS foundations from Module 1?"

**Expected Results**:
- System retrieves relevant content from multiple modules
- Responses synthesize information from different parts of the book
- Source citations indicate content from multiple modules
- Responses maintain accuracy across module boundaries

### Scenario 5: Edge Case Handling
**Objective**: Test system behavior with challenging queries

**Test Cases**:
1. Queries about content not in the book
2. Vague or ambiguous queries
3. Very specific technical queries
4. Queries that require synthesis of multiple concepts

**Expected Results**:
- System acknowledges when information is not in the book
- Vague queries return general information with clarification requests
- Specific queries return precise, relevant information
- Synthesis queries combine information from multiple sources

## Testing Scripts

### 1. Indexing Verification Script

```javascript
// test-indexing.js
const weaviate = require('weaviate-client');

const client = weaviate.client({
  scheme: process.env.WEAVIATE_SCHEME || 'http',
  host: process.env.WEAVIATE_HOST || 'localhost:8080',
});

async function testIndexing() {
  console.log('Testing content indexing...');
  
  try {
    // Count total indexed documents
    const result = await client.graphql
      .aggregate()
      .withClassName('DocChunk')
      .withFields('meta { count }')
      .do();
    
    const totalCount = result.data.Aggregate.DocChunk[0].meta.count;
    console.log(`Total indexed documents: ${totalCount}`);
    
    // Check for expected modules
    const modules = ['Module 1', 'Module 2', 'Module 3', 'Module 4'];
    for (const module of modules) {
      const moduleResult = await client.graphql
        .aggregate()
        .withClassName('DocChunk')
        .withWhere({
          path: ['module'],
          operator: 'Equal',
          valueString: module
        })
        .withFields('meta { count }')
        .do();
      
      const moduleCount = moduleResult.data.Aggregate.DocChunk[0].meta.count;
      console.log(`${module}: ${moduleCount} chunks`);
    }
    
    // Sample a few documents to verify content quality
    const sampleResult = await client.graphql
      .get()
      .withClassName('DocChunk')
      .withFields('content sourceDocument module chapter')
      .withLimit(5)
      .do();
    
    console.log('\nSample indexed content:');
    sampleResult.data.Get.DocChunk.forEach((doc, i) => {
      console.log(`${i+1}. Module: ${doc.module}`);
      console.log(`   Chapter: ${doc.chapter}`);
      console.log(`   Source: ${doc.sourceDocument}`);
      console.log(`   Content preview: ${doc.content.substring(0, 100)}...`);
      console.log('   ---');
    });
    
    return totalCount > 0;
  } catch (error) {
    console.error('Indexing test failed:', error);
    return false;
  }
}

module.exports = { testIndexing };
```

### 2. Search Functionality Test Script

```javascript
// test-search.js
const axios = require('axios');

async function testSearchFunctionality() {
  console.log('Testing search functionality...');
  
  const testQueries = [
    { query: "What is ROS 2 middleware?", expectedModule: "Module 1" },
    { query: "Isaac Sim setup process", expectedModule: "Module 3" },
    { query: "Nav2 path planning humanoid", expectedModule: "Module 3" },
    { query: "VSLAM in Isaac ROS", expectedModule: "Module 3" },
    { query: "URDF for humanoids", expectedModule: "Module 1" }
  ];
  
  let passedTests = 0;
  
  for (const test of testQueries) {
    try {
      console.log(`\nTesting query: "${test.query}"`);
      
      const response = await axios.post('http://localhost:3001/api/search', {
        query: test.query,
        limit: 3
      });
      
      if (response.data.results && response.data.results.length > 0) {
        // Check if the first result is from the expected module
        const firstResult = response.data.results[0];
        const hasExpectedModule = firstResult.module.includes(test.expectedModule);
        
        console.log(`  ✓ Found ${response.data.results.length} results`);
        console.log(`  ✓ First result from: ${firstResult.module}`);
        console.log(`  ✓ Relevance: ${firstResult.similarity.toFixed(2)}`);
        
        if (hasExpectedModule) {
          console.log(`  ✓ Correct module identified`);
          passedTests++;
        } else {
          console.log(`  ⚠ Module mismatch (expected ${test.expectedModule})`);
        }
      } else {
        console.log(`  ✗ No results returned`);
      }
    } catch (error) {
      console.log(`  ✗ Error: ${error.message}`);
    }
  }
  
  console.log(`\nSearch tests: ${passedTests}/${testQueries.length} passed`);
  return passedTests === testQueries.length;
}

module.exports = { testSearchFunctionality };
```

### 3. Chat Functionality Test Script

```javascript
// test-chat.js
const axios = require('axios');

async function testChatFunctionality() {
  console.log('Testing chat functionality...');
  
  try {
    // Create a new conversation
    const convResponse = await axios.post('http://localhost:3001/api/conversation', {
      initialMessage: 'I want to learn about ROS 2 concepts'
    });
    
    const conversationId = convResponse.data.conversationId;
    console.log(`Created conversation: ${conversationId}`);
    
    // Test conversation flow
    const conversationSteps = [
      { query: "What is ROS 2?", expectedTopic: "ROS 2" },
      { query: "How do nodes communicate?", expectedTopic: "nodes" },
      { query: "What are services?", expectedTopic: "services" }
    ];
    
    let passedTests = 0;
    
    for (const step of conversationSteps) {
      console.log(`\nStep: ${step.query}`);
      
      const response = await axios.post('http://localhost:3001/api/chat', {
        message: step.query,
        conversationId: conversationId
      });
      
      if (response.data.response) {
        console.log(`  ✓ Response received (${response.data.response.length} chars)`);
        
        // Check if response contains expected topic
        const containsTopic = response.data.response.toLowerCase().includes(step.expectedTopic.toLowerCase());
        if (containsTopic) {
          console.log(`  ✓ Contains expected topic: ${step.expectedTopic}`);
          passedTests++;
        } else {
          console.log(`  ⚠ May not contain expected topic: ${step.expectedTopic}`);
        }
        
        // Check for sources
        if (response.data.sources && response.data.sources.length > 0) {
          console.log(`  ✓ Sources provided: ${response.data.sources.length}`);
        } else {
          console.log(`  ⚠ No sources provided`);
        }
      } else {
        console.log(`  ✗ No response received`);
      }
    }
    
    console.log(`\nChat tests: ${passedTests}/${conversationSteps.length} passed`);
    return passedTests >= conversationSteps.length - 1; // Allow 1 failure
    
  } catch (error) {
    console.error('Chat test failed:', error.message);
    return false;
  }
}

module.exports = { testChatFunctionality };
```

### 4. Comprehensive Test Suite

```javascript
// test-rag-comprehensive.js
const { testIndexing } = require('./test-indexing');
const { testSearchFunctionality } = require('./test-search');
const { testChatFunctionality } = require('./test-chat');

async function runComprehensiveTests() {
  console.log('Starting comprehensive RAG functionality tests...\n');
  
  const tests = [
    { name: 'Content Indexing', fn: testIndexing },
    { name: 'Search Functionality', fn: testSearchFunctionality },
    { name: 'Chat Functionality', fn: testChatFunctionality }
  ];
  
  const results = [];
  
  for (const test of tests) {
    console.log(`\n${'='.repeat(50)}`);
    console.log(`Running: ${test.name}`);
    console.log(`${'='.repeat(50)}`);
    
    try {
      const result = await test.fn();
      results.push({ name: test.name, passed: result });
      console.log(`\n${test.name}: ${result ? 'PASSED' : 'FAILED'}`);
    } catch (error) {
      console.error(`\n${test.name}: ERROR - ${error.message}`);
      results.push({ name: test.name, passed: false, error: error.message });
    }
  }
  
  // Summary
  console.log(`\n${'='.repeat(50)}`);
  console.log('TEST SUMMARY');
  console.log(`${'='.repeat(50)}`);
  
  let passedCount = 0;
  for (const result of results) {
    const status = result.passed ? '✓ PASS' : '✗ FAIL';
    console.log(`${result.name}: ${status}`);
    if (result.passed) passedCount++;
  }
  
  console.log(`\nOverall: ${passedCount}/${results.length} tests passed`);
  
  const allPassed = passedCount === results.length;
  console.log(`Final Result: ${allPassed ? 'SUCCESS' : 'FAILURE'}`);
  
  return allPassed;
}

// Run tests if this file is executed directly
if (require.main === module) {
  runComprehensiveTests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('Test suite failed:', error);
      process.exit(1);
    });
}

module.exports = { runComprehensiveTests };
```

## Performance Testing

### Load Testing Script

```javascript
// test-performance.js
const axios = require('axios');
const { performance } = require('perf_hooks');

async function testPerformance() {
  console.log('Running performance tests...');
  
  const queries = [
    "What is ROS 2 middleware?",
    "Explain Isaac Sim setup",
    "How does Nav2 work?",
    "What are perception networks?",
    "Explain URDF for humanoids"
  ];
  
  const results = [];
  
  for (let i = 0; i < 10; i++) { // Run each query 10 times
    for (const query of queries) {
      const startTime = performance.now();
      
      try {
        const response = await axios.post('http://localhost:3001/api/search', {
          query: query,
          limit: 3
        });
        
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        
        results.push({
          query,
          responseTime,
          resultCount: response.data.results.length,
          success: true
        });
        
        console.log(`Query: "${query.substring(0, 20)}...", Time: ${responseTime.toFixed(2)}ms`);
      } catch (error) {
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        
        results.push({
          query,
          responseTime,
          success: false,
          error: error.message
        });
        
        console.log(`Query: "${query.substring(0, 20)}...", Failed: ${error.message}`);
      }
    }
  }
  
  // Calculate statistics
  const successfulResults = results.filter(r => r.success);
  const responseTimes = successfulResults.map(r => r.responseTime);
  
  if (responseTimes.length > 0) {
    const avgTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
    const minTime = Math.min(...responseTimes);
    const maxTime = Math.max(...responseTimes);
    
    console.log(`\nPerformance Statistics:`);
    console.log(`Average response time: ${avgTime.toFixed(2)}ms`);
    console.log(`Min response time: ${minTime.toFixed(2)}ms`);
    console.log(`Max response time: ${maxTime.toFixed(2)}ms`);
    console.log(`Success rate: ${successfulResults.length}/${results.length}`);
    
    // Check if performance meets requirements
    const meetsPerformance = avgTime < 2000; // Less than 2 seconds
    console.log(`Performance target met: ${meetsPerformance ? 'YES' : 'NO'}`);
    
    return {
      meetsPerformance,
      avgTime,
      successRate: successfulResults.length / results.length
    };
  }
  
  return { meetsPerformance: false, avgTime: Infinity, successRate: 0 };
}

module.exports = { testPerformance };
```

## Accuracy Testing

### Response Accuracy Validation

```javascript
// test-accuracy.js
const axios = require('axios');

// Known facts from the book content to validate against
const KNOWN_FACTS = {
  "ROS 2": [
    "middleware for robot control",
    "DDS (Data Distribution Service)",
    "improved upon ROS 1",
    "real-time capable"
  ],
  "Isaac Sim": [
    "photorealistic simulation",
    "synthetic data generation",
    "NVIDIA platform",
    "compatible with Isaac ROS"
  ],
  "Nav2": [
    "navigation stack 2",
    "for mobile robots",
    "path planning",
    "costmap-based"
  ]
};

async function testResponseAccuracy() {
  console.log('Testing response accuracy...');
  
  const testQueries = [
    { query: "What is ROS 2?", topic: "ROS 2" },
    { query: "Explain Isaac Sim", topic: "Isaac Sim" },
    { query: "What is Nav2?", topic: "Nav2" }
  ];
  
  let accurateResponses = 0;
  
  for (const test of testQueries) {
    console.log(`\nTesting: ${test.query}`);
    
    try {
      const response = await axios.post('http://localhost:3001/api/chat', {
        message: test.query,
        conversationId: `test-${Date.now()}`
      });
      
      const responseText = response.data.response.toLowerCase();
      const expectedKeywords = KNOWN_FACTS[test.topic];
      
      let keywordMatches = 0;
      for (const keyword of expectedKeywords) {
        if (responseText.includes(keyword.toLowerCase())) {
          keywordMatches++;
        }
      }
      
      console.log(`  Found ${keywordMatches}/${expectedKeywords.length} expected keywords`);
      
      if (keywordMatches >= Math.ceil(expectedKeywords.length * 0.6)) { // 60% threshold
        console.log(`  ✓ Response likely accurate`);
        accurateResponses++;
      } else {
        console.log(`  ⚠ Response may lack accuracy`);
        console.log(`  Response preview: ${response.data.response.substring(0, 200)}...`);
      }
    } catch (error) {
      console.log(`  ✗ Error getting response: ${error.message}`);
    }
  }
  
  console.log(`\nAccuracy tests: ${accurateResponses}/${testQueries.length} passed`);
  return accurateResponses === testQueries.length;
}

module.exports = { testResponseAccuracy };
```

## Test Execution Plan

### 1. Pre-Test Setup
- Ensure all content is indexed in the vector database
- Verify API server is running and accessible
- Confirm all required environment variables are set

### 2. Sequential Test Execution
1. Run indexing verification
2. Execute search functionality tests
3. Test chat conversation flow
4. Perform performance testing
5. Validate response accuracy

### 3. Post-Test Analysis
- Generate test reports with pass/fail status
- Identify areas needing improvement
- Document performance metrics
- Log any issues found for resolution

This comprehensive testing approach ensures that the RAG functionality works correctly with the complete book content, providing users with accurate and relevant information retrieval.