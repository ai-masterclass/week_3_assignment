/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  async rewrites() {
    // Use environment variable for API URL, fallback to localhost for development
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8082';
    
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        source: '/docs',
        destination: `${apiUrl}/docs`,
      },
      {
        source: '/health',
        destination: `${apiUrl}/health`,
      },
    ];
  },
};

export default nextConfig;

