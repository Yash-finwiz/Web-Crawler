# Web Crawler

A powerful, asynchronous web crawler built with Python and Playwright for efficient and stealthy web crawling.

## Features

- Asynchronous crawling with Playwright for JavaScript-heavy websites
- MongoDB integration for storing results and maintaining crawl state
- Support for proxies and rotating user agents
- Respect for robots.txt
- Configurable crawl depth, timeouts, and limits
- Product page detection
- Rate limiting and error handling

## Requirements

- Python 3.7+
- MongoDB
- Required Python packages (see below)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install playwright bs4 tqdm requests pymongo pybloom-live robotexclusionrulesparser
   ```
3. Install Playwright browsers:
   ```
   playwright install
   ```

## Configuration

Edit the `config.json` file to customize crawler settings:

- `results_dir`: Where to store crawling results
- `target_domains`: List of domains to crawl
- `max_depth`: Maximum depth to crawl
- `max_urls`: Maximum number of URLs to process
- `mongodb_uri`: MongoDB connection string
- Additional settings for user agents, proxies, timeouts, etc.

## Usage

```
python craw.py
```

## Files

- `craw.py`: Main crawler code
- `config.json`: Configuration settings
- `user_agents.txt`: List of user agents for rotation
- `proxies.txt`: List of proxies for rotation
- `results/`: Directory where crawling results are stored 