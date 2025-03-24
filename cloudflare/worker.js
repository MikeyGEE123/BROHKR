// Cloudflare Worker for BROHKR Trading Platform

/**
 * This worker handles routing for the BROHKR Trading Platform
 * It routes API requests to the backend and serves static assets from Cloudflare Pages
 */

const API_ROUTES = [
  '/api/exchanges',
  '/api/price_chart',
  '/api/order_book',
  '/api/exchange_comparison',
  '/deployment-info'
];

// Configuration (replace with your actual backend URL when deployed)
const BACKEND_URL = 'https://your-backend-api.com';

// Handle incoming requests
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  const url = new URL(request.url);
  const path = url.pathname;
  
  // Check if this is an API request
  if (isApiRequest(path)) {
    return handleApiRequest(request, path);
  }
  
  // Otherwise, serve static assets from Cloudflare Pages
  return fetch(request);
}

function isApiRequest(path) {
  return API_ROUTES.some(route => path.startsWith(route));
}

async function handleApiRequest(request, path) {
  // Clone the request to modify it
  const apiRequest = new Request(BACKEND_URL + path, {
    method: request.method,
    headers: request.headers,
    body: request.body,
    redirect: 'follow'
  });
  
  try {
    // Forward the request to the backend API
    const response = await fetch(apiRequest);
    
    // Clone the response to modify it
    const newResponse = new Response(response.body, response);
    
    // Add CORS headers to allow cross-origin requests
    newResponse.headers.set('Access-Control-Allow-Origin', '*');
    newResponse.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    newResponse.headers.set('Access-Control-Allow-Headers', 'Content-Type');
    
    return newResponse;
  } catch (error) {
    // Return an error response if the API request fails
    return new Response(JSON.stringify({ error: 'API request failed' }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });
  }
}