#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kurdistan Universities News Crawler - Enhanced Version with Pagination
Scrapes news from Kurdish universities and saves to MySQL database.

Usage:
    python crawler.py              # scrape all universities
    python crawler.py --uni koya   # scrape only Koya University
    python crawler.py --list       # list available universities

Requirements:
    pip install requests beautifulsoup4 selenium mysql-connector-python python-dotenv lxml

Setup:
    1. Create .env file with your database credentials:
       DB_HOST=localhost
       DB_PORT=3306
       DB_USER=root
       DB_PASSWORD=your_password
       DB_NAME=university_website

    2. Install ChromeDriver for Selenium
    3. Run the script
"""

import os
import sys
import argparse
import time
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import logging
import hashlib

# Third-party imports
import requests
from bs4 import BeautifulSoup
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib3

# Disable SSL warnings (only for development)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'university_website'),
    'charset': 'utf8mb4'
}

# Enhanced configuration with pagination support
UNIVERSITIES = {
    'koya': {
        'name': 'Koya University',
        'urls': [
            'https://koyauniversity.org/events/event-grid-3-column',
            'https://koyauniversity.org/ku/node/445',
            'https://koyauniversity.org/ar/node/445'
        ],
        'selectors': {
            'news_items': ['.views-row', '.card', '.event-card', '.news-item', '.view-content .row > div', 'article'],
            'title': ['h3 a', 'h2 a', 'h4 a', '.card-title a', '.title a', '.entry-title a', '.event-title a', 'h3', 'h2', 'h4', '.field-content a'],
            'link': ['a[href]'],
            'pagination': ['.pager-next a', '.pagination .next a', 'a[rel="next"]', '.page-numbers.next']
        },
        'use_selenium': True,
        'max_pages': 10
    },
    'garmian': {
        'name': 'Garmian University',
        'urls': [
            'https://garmian.edu.krd/en/news',
            'https://garmian.edu.krd/ar/news',
            'https://garmian.edu.krd/news'
        ],
        'selectors': {
            'news_items': ['.news-item'],
            'title': ['.news-title a', 'h4.news-title a'],
            'link': ['.news-title a', 'a[href]'],
            'pagination': [
                '.pagination li.active + li a',
                '.pagination li a[rel="next"]',
                '.pagination li a[aria-label="Next"]'
            ]
        },
        'use_selenium': False,
        'max_pages': 10
    },
    'goizha': {
        'name': 'University of Goizha',
        'urls': [
            'https://uog.edu.iq/news/',
            'https://uog.edu.iq/ku/hawalakan/'
        ],
        'selectors': {
            'news_items': ['.news-item'],
            'title': ['.news-title a', 'h4.news-title a'],
            'link': ['.news-title a', 'a[href]'],
            'pagination': [
                '.pagination li.active + li a',
                '.pagination li a[rel="next"]',
                '.pagination li a[aria-label="Next"]'
            ]
        },
        'use_selenium': False,
        'max_pages': 10
    },
    'univsul': {
        'name': 'University of Sulaimani',
        'urls': [
            'https://univsul.edu.iq/ar/akhbar/',
            'https://univsul.edu.iq/ku/hawalakan/',
            'https://univsul.edu.iq/en/news/'
        ],
        'selectors': {
            'news_items': ['.post', 'article', '.news-card', '.post-item', '.news-item', '.card', '.entry', '.blog-post'],
            'title': ['h3 a', 'h2 a', 'h4 a', '.title a', '.post-title a', '.card-title a', 'h3', 'h2', 'h4', '.entry-title a'],
            'link': ['a[href]'],
            'pagination': ['.next-posts-link', '.pagination .next', 'a[rel="next"]', '.page-numbers.next', '.nav-previous a']
        },
        'use_selenium': True,
        'max_pages': 15
    },
    'qaiwan': {
        'name': 'Qaiwan International University',
        'urls': ['https://www.uniq.edu.iq/News'],
        'selectors': {
            'news_items': ['.single-news-card', '.news-card', '.latest-news-card', '.news-item', '.card', '.news-container > div'],
            'title': ['h5 a', 'h3 a', 'h4 a', '.title a', '.card-title a', 'h5', 'h3', 'h4', '.news-title a'],
            'link': ['a[href]'],
            'pagination': ['.pagination .next a', 'a[rel="next"]', '.page-numbers.next', '.load-more']
        },
        'use_selenium': True,
        'max_pages': 8
    },
    'spu': {
        'name': 'Sulaimani Polytechnic University',
        'urls': [
            'https://spu.edu.iq/en/category/services/news-activities',
            'https://spu.edu.iq/ku/?cat=87',
            'https://spu.edu.iq/ar/category/%d8%a7%d9%84%d9%85%d8%b7%d8%a8%d9%88%d8%b9%d8%a7%d8%aa/%d8%a7%d9%84%d8%a7%d9%86%d8%b4%d8%b7%d8%a9-%d9%88%d8%a7%d9%84%d8%a7%d8%ae%d8%a8%d8%a7%d8%b1/'
        ],
        'selectors': {
            'news_items': ['.post', 'article', '.entry', '.post-item', '.hentry', '.type-post'],
            'title': ['h2 a', 'h3 a', '.entry-title a', '.post-title a', 'h2', 'h3', '.title a'],
            'link': ['a[href]'],
            'pagination': ['.nav-previous a', '.next.page-numbers', 'a[rel="next"]', '.pagination .next a']
        },
        'use_selenium': False,
        'max_pages': 12
    },
    'salahaddin': {
        'name': 'Salahaddin University',
        'urls': ['https://su.edu.krd/ku/news'],
        'selectors': {
            'news_items': ['.view-content .views-row', '.news-item', 'article', '.post', '.card', '.news-card', '.content-item'],
            'title': ['h3 a', 'h2 a', '.title a', 'h3', 'h2', '.field-content a', '.views-field-title a'],
            'link': ['a[href]'],
            'pagination': ['.pager-next a', '.pagination .next a', 'a[rel="next"]', '.page-numbers.next']
        },
        'use_selenium': True,
        'max_pages': 10
    }
}

DELAY_BETWEEN_REQUESTS = 2
DELAY_BETWEEN_PAGES = 3
PAGE_LOAD_TIMEOUT = 90
MAX_ARTICLES_PER_PAGE = 400  # Increased limit

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crawler.log', mode='w', encoding='utf-8')
    ]
)

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def connect_database():
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        logging.info("Successfully connected to MySQL database")
        return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        return None

def create_table(connection):
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS sim_news (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        title TEXT NOT NULL,
        content LONGTEXT,
        link TEXT NOT NULL,
        language CHAR(2),
        url_hash VARCHAR(64) UNIQUE,
        source VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_link (link(255))
    )
    """
    try:
        cursor.execute(create_table_query)
        connection.commit()
        logging.info("Table 'sim_news' created/verified successfully")
        cursor.close()
        return True
    except Error as e:
        logging.error(f"Error creating table: {e}")
        cursor.close()
        return False

def generate_url_hash(url):
    """Generate a hash for the URL to handle duplicates better"""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def insert_article(connection, title, content, link, language, source):
    cursor = connection.cursor()
    url_hash = generate_url_hash(link)
    insert_query = """
    INSERT INTO sim_news (title, content, link, language, url_hash, source)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (title, content, link, language, url_hash, source))
        connection.commit()
        cursor.close()
        return True
    except mysql.connector.IntegrityError:
        cursor.close()
        return False
    except Error as e:
        logging.error(f"Error inserting article: {e}")
        cursor.close()
        return False

def clear_table(connection):
    """Clear all data from sim_news table for testing"""
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM sim_news")
        connection.commit()
        logging.info("Cleared all data from sim_news table")
        cursor.close()
        return True
    except Error as e:
        logging.error(f"Error clearing table: {e}")
        cursor.close()
        return False

# =============================================================================
# SCRAPING FUNCTIONS
# =============================================================================

def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--disable-logging')
    chrome_options.add_argument('--log-level=3')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins')
    chrome_options.add_argument('--disable-images')  # Speed up loading

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        return driver
    except Exception as e:
        logging.error(f"Error setting up Selenium: {e}")
        return None

def fetch_with_requests(url, verify_ssl=False):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    try:
        response = requests.get(url, headers=headers, timeout=30, verify=verify_ssl)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def fetch_with_selenium(driver, url):
    try:
        logging.info(f"Loading page: {url}")
        driver.get(url)

        # Wait for page to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            pass

        # Additional wait for dynamic content
        time.sleep(3)
        return driver.page_source
    except Exception as e:
        logging.error(f"Error fetching {url} with Selenium: {e}")
        return None

def detect_language(text):
    if not text:
        return 'en'
    kurdish_chars = ['ئ', 'ێ', 'ە', 'ڕ', 'ڵ', 'ۆ', 'ڤ', 'گ', 'چ', 'ژ']
    arabic_chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    kurdish_count = sum(1 for char in text if char in kurdish_chars)
    arabic_count = sum(1 for char in text if char in arabic_chars)
    if kurdish_count > 0:
        return 'ku'
    elif arabic_count > 0:
        return 'ar'
    else:
        return 'en'

def extract_article_data(item, base_url, selectors):
    """Extract title and link from news item with improved logic"""
    title = ''
    link = ''

    # Try to find title
    for selector in selectors['title']:
        title_elem = item.select_one(selector)
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title and len(title) > 3:
                break

    # Try to find link
    for selector in selectors['link']:
        link_elem = item.select_one(selector)
        if link_elem and link_elem.get('href'):
            href = link_elem.get('href')
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                link = urljoin(base_url, href)
                break

    # If no link found in selectors, try to find any link in the item
    if not link:
        all_links = item.find_all('a', href=True)
        for a_tag in all_links:
            href = a_tag.get('href')
            if href and not href.startswith('#') and not href.startswith('javascript:') and not href.startswith('mailto:'):
                link = urljoin(base_url, href)
                break

    # If no title found, try to get it from the link text
    if not title and link:
        link_elem = item.find('a', href=True)
        if link_elem:
            title = link_elem.get_text(strip=True)

    return title, link

def get_all_pages(start_url, config, driver=None):
    """Generator that yields all pages with pagination support"""
    visited_urls = set()
    urls_to_visit = [start_url]
    page_count = 0
    max_pages = config.get('max_pages', 5)

    while urls_to_visit and page_count < max_pages:
        current_url = urls_to_visit.pop(0)

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        page_count += 1

        logging.info(f"Fetching page {page_count}: {current_url}")

        # Fetch the page
        if config['use_selenium'] and driver:
            html = fetch_with_selenium(driver, current_url)
        else:
            html = fetch_with_requests(current_url, verify_ssl=False)

        if not html:
            continue

        soup = BeautifulSoup(html, 'html.parser')
        yield soup, current_url

        # Look for next page link
        next_url = None
        for pagination_selector in config['selectors'].get('pagination', []):
            next_elem = soup.select_one(pagination_selector)
            if next_elem and next_elem.get('href'):
                next_url = urljoin(current_url, next_elem['href'])
                break

        # Add next URL if found and not visited
        if next_url and next_url not in visited_urls:
            # Avoid infinite loops by checking if URL is reasonable
            if urlparse(next_url).netloc == urlparse(current_url).netloc:
                urls_to_visit.append(next_url)
                logging.info(f"Found next page: {next_url}")

        # Delay between pages
        if urls_to_visit:
            time.sleep(DELAY_BETWEEN_PAGES)

def fetch_article_content(url, use_selenium=False, driver=None):
    """Fetch full article content from article page"""
    try:
        if use_selenium and driver:
            html = fetch_with_selenium(driver, url)
        else:
            html = fetch_with_requests(url, verify_ssl=False)

        if not html:
            return ''

        soup = BeautifulSoup(html, 'html.parser')
        content_selectors = [
            '.post-content', '.entry-content', '.article-content',
            '.content', 'main p', 'article p', '.post p', '.news-content',
            '.field-type-text-with-summary', '.field-name-body'
        ]

        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(' ', strip=True)
                if len(content) > 50:  # Only return if substantial content
                    return content[:5000]

        # Fallback to first few paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.get_text(' ', strip=True) for p in paragraphs[:3]])
            return content[:2000] if content else ''

        return ''
    except Exception as e:
        logging.error(f"Error fetching content from {url}: {e}")
        return ''

def scrape_university(uni_key, connection):
    if uni_key not in UNIVERSITIES:
        logging.warning(f"University '{uni_key}' not found")
        return 0

    config = UNIVERSITIES[uni_key]
    logging.info(f"\n{'='*60}")
    logging.info(f"Scraping {config['name']}")
    logging.info(f"{'='*60}")

    driver = None
    if config['use_selenium']:
        driver = setup_selenium()
        if not driver:
            logging.warning("Failed to setup Selenium")
            return 0

    total_saved = 0

    try:
        for base_url in config['urls']:
            logging.info(f"\nProcessing base URL: {base_url}")

            page_count = 0
            for soup, page_url in get_all_pages(base_url, config, driver):
                page_count += 1
                logging.info(f"Processing page {page_count}: {page_url}")

                # Find news items
                news_items = []
                for selector in config['selectors']['news_items']:
                    news_items = soup.select(selector)
                    if news_items:
                        logging.info(f"Found {len(news_items)} items with selector '{selector}'")
                        break

                if not news_items:
                    logging.warning(f"No news items found on page {page_count}")
                    continue

                saved_count = 0
                for i, item in enumerate(news_items[:MAX_ARTICLES_PER_PAGE]):
                    try:
                        title, link = extract_article_data(item, page_url, config['selectors'])

                        if not title or not link:
                            continue

                        # Clean and validate title
                        title = title.strip()
                        if len(title) < 5:
                            continue

                        # Skip if link is not a proper article link
                        if any(skip in link.lower() for skip in ['#', 'javascript:', 'mailto:', '.pdf', '.doc']):
                            continue

                        language = detect_language(title)

                        # Fetch article content (with rate limiting)
                        content = ''
                        try:
                            content = fetch_article_content(link, use_selenium=config['use_selenium'], driver=driver)
                            time.sleep(1)  # Rate limiting for content fetching
                        except Exception as e:
                            logging.warning(f"Could not fetch content for {link}: {e}")

                        success = insert_article(connection, title, content, link, language, config['name'])
                        if success:
                            saved_count += 1
                            total_saved += 1
                            logging.info(f"✓ Saved: {title[:80]}...")
                        else:
                            logging.debug(f"⚠ Skipped (duplicate): {title[:80]}...")

                    except Exception as e:
                        logging.error(f"Error processing item {i+1}: {e}")
                        continue

                logging.info(f"Saved {saved_count} new articles from page {page_count}")
                time.sleep(DELAY_BETWEEN_REQUESTS)

    except Exception as e:
        logging.error(f"Error scraping {config['name']}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

    logging.info(f"\nTotal saved for {config['name']}: {total_saved} articles")
    return total_saved

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def scrape_all_universities(connection):
    logging.info("Starting to scrape all universities...")
    total_saved = 0
    results = {}

    for uni_key in UNIVERSITIES.keys():
        try:
            saved = scrape_university(uni_key, connection)
            results[uni_key] = saved
            total_saved += saved
        except Exception as e:
            logging.error(f"Error scraping {uni_key}: {e}")
            results[uni_key] = 0

    logging.info(f"\n{'='*60}")
    logging.info("FINAL SUMMARY")
    logging.info(f"{'='*60}")

    for uni_key, saved in results.items():
        uni_name = UNIVERSITIES[uni_key]['name']
        logging.info(f"• {uni_name}: {saved} new articles")

    logging.info(f"\nTotal articles saved: {total_saved}")
    return total_saved

def list_universities():
    print("Available universities:")
    print("-" * 40)
    for key, config in UNIVERSITIES.items():
        print(f"[{key}] {config['name']}")
        for url in config['urls']:
            print(f"   - {url}")
        print(f"   Max pages: {config.get('max_pages', 5)}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Kurdistan Universities News Crawler - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--uni', '--university',
                       choices=['all'] + list(UNIVERSITIES.keys()),
                       default='all',
                       help='University to scrape (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available universities')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all data from database before scraping')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with more verbose logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list:
        list_universities()
        return

    if not os.path.exists('.env'):
        print("Error: .env file not found!")
        print("Create a .env file with your database credentials:")
        print("DB_HOST=localhost")
        print("DB_PORT=3306")
        print("DB_USER=root")
        print("DB_PASSWORD=your_password")
        print("DB_NAME=university_website")
        return

    connection = connect_database()
    if not connection:
        return

    if not create_table(connection):
        connection.close()
        return

    if args.clear:
        if not clear_table(connection):
            connection.close()
            return

    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.uni == 'all':
            total_saved = scrape_all_universities(connection)
        else:
            total_saved = scrape_university(args.uni, connection)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print(f"Total articles saved: {total_saved}")
        logging.info(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Duration: {duration}")
        logging.info(f"Total articles saved: {total_saved}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logging.warning("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception(f"\nError: {e}")
    finally:
        connection.close()
        print("Database connection closed")
        logging.info("Database connection closed")

if __name__ == '__main__':
    main()

