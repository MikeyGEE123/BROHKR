# BROHKR Trading Platform - Cloudflare Deployment Guide

## Overview

This guide will help you deploy the BROHKR Trading Platform to Cloudflare Pages and set up a custom domain.

## Prerequisites

1. A Cloudflare account
2. A domain name registered with Cloudflare (or transferred to Cloudflare DNS)
3. The BROHKR Trading Platform codebase

## Deployment Steps

### 1. Set Up Cloudflare Pages

1. Log in to your Cloudflare dashboard at https://dash.cloudflare.com
2. Navigate to "Pages" from the sidebar
3. Click "Create a project"
4. Choose "Connect to Git" and connect your GitHub/GitLab repository containing the BROHKR codebase
5. Configure your build settings:
   - Build command: `pip install -r requirements.txt && python -m flask run --port 8080`
   - Build output directory: `web/static`
   - Root directory: `/`

### 2. Configure Environment Variables

In your Cloudflare Pages project settings, add the following environment variables:

- `FLASK_ENV`: Set to `production` for production deployment
- Any API keys or secrets needed by your application (use encrypted variables for sensitive information)

### 3. Set Up Custom Domain

1. In your Cloudflare Pages project, go to "Custom domains"
2. Click "Set up a custom domain"
3. Enter your domain name (e.g., `trading.yourdomain.com`)
4. Follow the verification steps if your domain is not already on Cloudflare
5. Choose the appropriate SSL/TLS encryption mode (Full recommended)

### 4. Configure DNS Settings

1. Go to the DNS section of your Cloudflare dashboard
2. Add a CNAME record:
   - Type: CNAME
   - Name: trading (or your subdomain of choice)
   - Target: your-pages-project.pages.dev
   - Proxy status: Proxied (recommended)

### 5. Advanced Configuration (Optional)

#### Custom Workers

If you need additional functionality beyond static hosting, you can create Cloudflare Workers to handle dynamic requests:

1. Create a `workers-site` directory in your project
2. Add your Worker scripts
3. Update the `wrangler.toml` configuration file

#### API Routing

To handle API requests, you can set up Workers to proxy requests to your backend:

```js
// Example Worker script for API routing
addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  
  if (url.pathname.startsWith('/api/')) {
    // Proxy to your API backend
    return fetch('https://your-api-backend.com' + url.pathname);
  }
  
  // Serve static assets
  return fetch(event.request);
});
```

## Troubleshooting

### Common Issues

1. **Build Failures**: Check your build logs in the Cloudflare Pages dashboard
2. **Domain Not Connecting**: Verify DNS settings and ensure your domain is properly configured in Cloudflare
3. **API Requests Failing**: Check CORS settings and ensure your Workers are properly routing requests

### Support Resources

- Cloudflare Pages Documentation: https://developers.cloudflare.com/pages/
- Cloudflare Workers Documentation: https://developers.cloudflare.com/workers/
- Cloudflare Community Forum: https://community.cloudflare.com/

## Maintenance

After deployment, monitor your application's performance using Cloudflare Analytics. Regular updates can be deployed by pushing changes to your connected Git repository.