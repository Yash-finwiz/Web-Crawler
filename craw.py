import asyncio
import re
import csv
import time
import os
import random
import logging
import json
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
from pymongo import MongoClient
from pybloom_live import ScalableBloomFilter
from robotexclusionrulesparser import RobotExclusionRulesParser

# Configuration files
CONFIG_FILE = 'config.json'
USER_AGENTS_FILE = 'user_agents.txt'
PROXIES_FILE = 'proxies.txt'

# Load configuration
with open(CONFIG_FILE) as f:
    config = json.load(f)

# Initialize logging
logging.basicConfig(
    level=config.get('log_level', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedCrawler:
    def __init__(self, target_domains):
        self.target_domains = target_domains
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load rotating user agents and proxies
        self.user_agents = self._load_list_file(USER_AGENTS_FILE)
        self.proxies = self._load_list_file(PROXIES_FILE)
        self.current_proxy_idx = 0
        
        # Initialize Bloom filter with larger capacity
        self.visited_urls = ScalableBloomFilter(
            mode=ScalableBloomFilter.LARGE_SET_GROWTH,
            initial_capacity=10000
        )
        
        # Initialize MongoDB connection with connection pooling
        self._init_database()
        
        # Initialize robots.txt parser
        self.robot_parser = RobotExclusionRulesParser()
        
        # Track stats
        self.stats = {
            'urls_processed': 0,
            'products_found': 0,
            'start_time': time.time()
        }
        
    def _load_list_file(self, filename):
        """Load a text file into a list"""
        try:
            with open(filename) as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []
            
    def _init_database(self):
        """Initialize MongoDB database with optimized connection"""
        try:
            # Connect to MongoDB with optimized settings
            self.client = MongoClient(
                config.get('mongodb_uri', 'mongodb://localhost:27017/'),
                maxPoolSize=50,
                connectTimeoutMS=5000,
                socketTimeoutMS=30000,
                serverSelectionTimeoutMS=5000,
                w=1  # Faster write concern
            )
            self.db = self.client[config.get('mongodb_db', 'crawler_db')]
            
            # Create collections if they don't exist
            self.urls_collection = self.db['urls']
            self.crawl_state_collection = self.db['crawl_state']
            
            # Create indexes for faster queries
            self.urls_collection.create_index([('url', 1)], unique=True)
            self.urls_collection.create_index([('domain', 1)])
            self.urls_collection.create_index([('is_product', 1)])
            self.crawl_state_collection.create_index([('domain', 1)], unique=True)
            
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        
    def _get_random_user_agent(self):
        """Get a random user agent from the list"""
        return random.choice(self.user_agents) if self.user_agents else config['default_user_agent']
        
    def _get_next_proxy(self):
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_proxy_idx % len(self.proxies)]
        self.current_proxy_idx += 1
        return {
            'server': proxy,
            'username': config.get('proxy_user'),
            'password': config.get('proxy_pass')
        }
        
    async def _check_robots_txt(self, domain):
        """Check robots.txt for crawling permissions"""
        try:
            robots_url = urljoin(domain, '/robots.txt')
            response = requests.get(robots_url, timeout=10)
            self.robot_parser.parse(response.text)
            return self.robot_parser.is_allowed(config['user_agent_name'], domain)
        except Exception as e:
            logger.warning(f"Failed to fetch robots.txt for {domain}: {str(e)}")
            return True  # Assume allowed if unavailable
            
    async def _save_crawl_state(self, domain, queue, in_progress):
        """Save current crawl state to MongoDB with bulk operations"""
        try:
            # Convert set to list for MongoDB serialization
            in_progress_list = list(in_progress) if isinstance(in_progress, set) else in_progress
            
            # Only save a limited number of URLs to avoid massive documents
            queue_limit = config.get('queue_save_limit', 1000)
            if len(queue) > queue_limit:
                queue = queue[:queue_limit]
            
            # Upsert the crawl state document
            self.crawl_state_collection.update_one(
                {'domain': domain},
                {'$set': {
                    'queue': queue,
                    'in_progress': in_progress_list,
                    'updated_at': time.time(),
                    'stats': self.stats
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save crawl state: {str(e)}")
        
    async def _load_crawl_state(self, domain):
        """Load previous crawl state from MongoDB"""
        try:
            state = self.crawl_state_collection.find_one({'domain': domain})
            if state:
                queue = state.get('queue', [])
                # Convert list back to set for in_progress
                in_progress = set(state.get('in_progress', []))
                return queue, in_progress
        except Exception as e:
            logger.error(f"Failed to load crawl state: {str(e)}")
        
        return None, None
        
    async def _random_delay(self, base=1.0, variance=0.5):
        """Add random delay between requests"""
        delay = base + random.uniform(-variance, variance)
        await asyncio.sleep(max(delay, 0.5))
        
    async def _stealth_navigation(self, page):
        """Perform stealth actions to avoid detection"""
        # Random mouse movements
        await page.mouse.move(
            random.randint(0, 1920),
            random.randint(0, 1080)
        )
            
        # Random keyboard input
        if random.random() < 0.3:
            await page.keyboard.press('PageDown')
            
        # Random viewport size
        await page.set_viewport_size({
            'width': random.choice([1920, 1366, 1440, 1600]),
            'height': random.choice([1080, 768, 900, 1024])
        })
        
    async def _handle_errors(self, response, url):
        """Handle HTTP errors with appropriate actions"""
        if response.status in [429, 503]:
            retry_after = response.headers.get('Retry-After', 30)
            logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
            await asyncio.sleep(int(retry_after))
            return 'retry'
        elif response.status >= 500:
            logger.error(f"Server error on {url}")
            return 'abort'
        return 'continue'
        
    async def crawl_domain(self, domain):
        """Enhanced domain crawling with state management and performance optimizations"""
        # Check robots.txt first
        logger.info(f"Checking robots.txt for {domain}")
        if not await self._check_robots_txt(domain):
            logger.warning(f"Skipping {domain} due to robots.txt restrictions")
            return
            
        # Load previous state if exists
        logger.info(f"Loading previous crawl state for {domain}")
        queue, in_progress = await self._load_crawl_state(domain)
        if queue is None:
            logger.info(f"No previous state found, starting fresh for {domain}")
            queue = [(domain, 0)]
            in_progress = set()
        else:
            logger.info(f"Continuing previous crawl with {len(queue)} URLs in queue")
            # Ensure in_progress is a set
            in_progress = set(in_progress) if not isinstance(in_progress, set) else in_progress
            
        # Set max depth from config
        max_depth = config.get('max_depth', 3)
        
        logger.info(f"Launching browser with proxy: {self._get_next_proxy()}")
        browser_options = {
            'headless': True,
            'proxy': self._get_next_proxy(),
            'args': config['browser_args']
        }
        
        # Add timeout for browser launch
        try:
            async with async_playwright() as p:
                browser = await asyncio.wait_for(
                    p.chromium.launch(**browser_options),
                    timeout=60
                )
                
                try:
                    start_time = time.time()
                    max_urls = config.get('max_urls', 1000)
                    
                    while queue and self.stats['urls_processed'] < max_urls:
                        # Process URLs in batches
                        batch = []
                        for i in range(min(config['max_concurrent_pages'], len(queue))):
                            if i >= len(queue):
                                break
                            url, depth = queue[i]
                            
                            # Skip URLs beyond max depth
                            if depth > max_depth:
                                continue
                                
                            batch.append((url, depth))
                        
                        if not batch:
                            break
                            
                        # Remove processed URLs from queue
                        del queue[:len(batch)]
                        
                        logger.info(f"Processing batch of {len(batch)} URLs (processed so far: {self.stats['urls_processed']})")
                        tasks = []
                        for url, depth in batch:
                            logger.debug(f"Creating new context for {url} at depth {depth}")
                            context = await browser.new_context(
                                user_agent=self._get_random_user_agent(),
                                **config['browser_context_args']
                            )
                            page = await context.new_page()
                            tasks.append(self.process_page(url, depth, page, context))
                            
                        logger.debug("Waiting for batch processing to complete")
                        results = await asyncio.gather(*tasks)
                        
                        # Process results and update queue
                        for result in results:
                            if result and result['new_urls']:
                                # Get URLs from result, filter duplicates, and prioritize product pages
                                filtered_urls = []
                                for new_url, new_depth in result['new_urls']:
                                    # Skip already queued URLs
                                    if any(new_url == queued_url for queued_url, _ in queue):
                                        continue
                                    
                                    # Skip beyond max depth
                                    if new_depth > max_depth:
                                        continue
                                        
                                    filtered_urls.append((new_url, new_depth))
                                    
                                logger.debug(f"Found {len(filtered_urls)} new URLs to crawl after filtering")
                                queue.extend(filtered_urls)
                                
                        # Update stats
                        self.stats['urls_processed'] += len(batch)
                        elapsed = time.time() - start_time
                        
                        # Report progress
                        if self.stats['urls_processed'] % 10 == 0:
                            logger.info(f"Progress: {self.stats['urls_processed']} URLs processed, "
                                       f"{self.stats['products_found']} products found, "
                                       f"Queue size: {len(queue)}, "
                                       f"Elapsed: {elapsed:.2f}s")
                                
                        # Save state periodically
                        if len(queue) % config['state_save_interval'] == 0:
                            logger.debug(f"Saving state with {len(queue)} URLs in queue")
                            await self._save_crawl_state(domain, queue, in_progress)
                            
                        # Check if we've hit time limit
                        if time.time() - start_time > config.get('max_runtime', 3600):
                            logger.info(f"Max runtime reached after {time.time() - start_time:.2f}s")
                            break
                            
                finally:
                    logger.info("Closing browser and saving final state")
                    await browser.close()
                    await self._save_crawl_state(domain, queue, in_progress)
        except Exception as e:
            logger.error(f"Browser error: {str(e)}")
                
    async def process_page(self, url, depth, page, context):
        """Process individual page with enhanced error handling and performance optimizations"""
        result = {'new_urls': []}
        try:
            # Extract domain from URL for later use
            domain = urlparse(url).netloc
            
            # Set a stricter navigation timeout to avoid hanging on slow pages
            navigation_timeout = config.get('navigation_timeout', 30000)
            
            # Navigate with exponential backoff
            for attempt in range(config['max_retries']):
                try:
                    # Set stricter timeout for slow sites
                    response = await asyncio.wait_for(
                        page.goto(
                            url,
                            timeout=navigation_timeout,
                            wait_until='domcontentloaded'
                        ),
                        timeout=navigation_timeout/1000 + 5  # Add buffer
                    )
                    
                    # Error handling
                    if response.status >= 400:
                        action = await self._handle_errors(response, url)
                        if action == 'retry':
                            continue
                        elif action == 'abort':
                            return result
                            
                    # Perform stealth actions
                    await self._stealth_navigation(page)
                    
                    # Extract content and links
                    content = await page.content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Save product URLs
                    if self.is_product_page(soup, url):
                        self._store_url(url, domain, is_product=True)
                        self.stats['products_found'] += 1
                        
                    # Extract and filter links - use more aggressive filtering for better performance
                    new_urls = []
                    if depth < config.get('max_depth', 3):  # Only extract links if not at max depth
                        links = self.extract_links(soup, url)
                        # Use Bloom filter for efficient duplicate filtering
                        for link in links:
                            # Add to new URLs if not visited
                            if not self.visited_urls.add(link):
                                parsed_link = urlparse(link)
                                
                                # Skip common non-product paths
                                skip_paths = ['/login', '/cart', '/checkout', '/account', '/wishlist', '/search']
                                if any(parsed_link.path.startswith(path) for path in skip_paths):
                                    continue
                                    
                                # Skip image URLs
                                if any(ext in parsed_link.path.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                                    continue
                                    
                                # Skip social links
                                if any(social in parsed_link.netloc.lower() for social in ['facebook', 'twitter', 'instagram', 'youtube']):
                                    continue
                                    
                                new_urls.append((link, depth+1))
                                
                    result['new_urls'] = new_urls
                    break
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout occurred for {url} on attempt {attempt+1}")
                    if attempt == config['max_retries'] - 1:
                        return result
                    await self._random_delay(base=2**attempt)
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == config['max_retries'] - 1:
                        return result
                    await self._random_delay(base=2**attempt)
        finally:
            await context.close()
            
        return result
        
    def _store_url(self, url, domain, is_product):
        """Store URL in MongoDB"""
        try:
            self.urls_collection.update_one(
                {'url': url},
                {'$set': {
                    'domain': domain,
                    'is_product': is_product,
                    'discovered_at': time.time()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            
    def is_product_page(self, soup, url):
        """Determine if the page is a product page based on common patterns"""
        # Check URL patterns
        url_patterns = ['/product/', '/p/', '/item/', '/products/']
        if any(pattern in url for pattern in url_patterns):
            return True
        
        # Check for common product page elements
        product_indicators = [
            soup.find('div', {'class': lambda x: x and 'product' in x.lower() if x else False}),
            soup.find('span', {'class': lambda x: x and 'price' in x.lower() if x else False}),
            soup.find('button', {'class': lambda x: x and ('cart' in x.lower() or 'buy' in x.lower()) if x else False}),
            soup.find('div', {'id': lambda x: x and 'product' in x.lower() if x else False})
        ]
        
        return any(product_indicators)
        
    def extract_links(self, soup, base_url):
        """Extract and filter links from the page with improved efficiency"""
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        # Pre-filter to reduce processing - only look at a subset of links
        links = []
        link_count = 0
        max_links = config.get('max_links_per_page', 100)
        
        for a_tag in soup.find_all('a', href=True, limit=max_links*2):
            if link_count >= max_links:
                break
            
            href = a_tag['href']
            if not href or href == '#' or href.startswith('javascript:'):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            
            # Skip fragment links, mailto, etc.
            if not parsed_url.scheme.startswith('http'):
                continue
            
            # Skip external domains if not in target list
            if parsed_url.netloc != base_domain and parsed_url.netloc not in self.target_domains:
                continue
            
            # Skip file downloads and non-HTML resources
            skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.zip', '.rar', '.exe', '.webp', '.mp4', '.mp3']
            if any(parsed_url.path.lower().endswith(ext) for ext in skip_extensions):
                continue
            
            # Clean the URL (remove tracking parameters, fragments, etc.)
            clean_url = parsed_url._replace(fragment='').geturl()
            
            # Skip parameters that indicate non-useful pages
            if any(param in clean_url for param in ['utm_', 'ref=', 'source=']):
                clean_url = clean_url.split('?')[0]
            
            # Remove trailing slash for consistency
            if clean_url.endswith('/'):
                clean_url = clean_url[:-1]
            
            # Check if allowed by robots.txt
            if not self.robot_parser.is_allowed(config['user_agent_name'], clean_url):
                continue
            
            links.append(clean_url)
            link_count += 1
        
        return links

async def main():
    logger.info("Starting crawler...")
    crawler = EnhancedCrawler(config['target_domains'])
    for domain in config['target_domains']:
        logger.info(f"Beginning to crawl domain: {domain}")
        await crawler.crawl_domain(domain)
    logger.info("Crawling completed.")
        
if __name__ == "__main__":
    asyncio.run(main())