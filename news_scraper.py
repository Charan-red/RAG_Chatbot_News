import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import os

class NewsCollector:
    def __init__(self, sources_config):
        """
        Initialize with a list of news sources and their configurations
        
        sources_config should be a list of dictionaries, each containing:
        - name: name of the news source
        - url: base URL for the news source
        - article_selector: CSS selector for article elements
        - title_selector: CSS selector for article titles
        - content_selector: CSS selector for article content
        - date_selector: CSS selector for article dates
        """
        self.sources = sources_config
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
    def scrape_all_sources(self):
        """Scrape all configured news sources"""
        all_articles = []
        
        for source in self.sources:
            print(f"Scraping articles from {source['name']}...")
            try:
                articles = self.scrape_source(source)
                all_articles.extend(articles)
                print(f"Successfully scraped {len(articles)} articles from {source['name']}")
            except Exception as e:
                print(f"Error scraping {source['name']}: {str(e)}")
                
        # Save all collected articles
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"data/news_articles_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=4)
            
        return all_articles
                
    def scrape_source(self, source):
        """Scrape a single news source based on its configuration"""
        articles = []
        
        try:
            response = requests.get(source['url'], headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            article_elements = soup.select(source['article_selector'])
            
            for element in article_elements[:10]:  # Limit to 10 articles per source
                try:
                    # Extract article URL
                    article_url = None
                    if 'link_selector' in source:
                        link_element = element.select_one(source['link_selector'])
                        if link_element and link_element.has_attr('href'):
                            article_url = link_element['href']
                            # Handle relative URLs
                            if article_url.startswith('/'):
                                base_url = '/'.join(source['url'].split('/')[:3])
                                article_url = base_url + article_url
                    
                    # Extract title
                    title = element.select_one(source['title_selector'])
                    title_text = title.text.strip() if title else "No title found"
                    
                    # Extract date if available
                    date = None
                    if 'date_selector' in source:
                        date_element = element.select_one(source['date_selector'])
                        if date_element:
                            date = date_element.text.strip()
                    
                    # For full content, we need to visit the article page
                    content_text = "Summary not available"
                    if article_url:
                        content_text = self.get_article_content(article_url, source.get('content_selector'))
                    
                    articles.append({
                        'source': source['name'],
                        'title': title_text,
                        'url': article_url,
                        'date': date,
                        'content': content_text,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Be polite and don't hammer the server
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing article: {str(e)}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {source['url']}: {str(e)}")
            
        return articles
        
    def get_article_content(self, url, content_selector):
        """Visit the article page and extract the content"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            content_element = soup.select_one(content_selector)
            
            if content_element:
                # Remove script and style elements
                for script in content_element(["script", "style"]):
                    script.extract()
                
                # Get text and clean it up
                content = content_element.text.strip()
                # Remove extra whitespace
                content = ' '.join(content.split())
                return content
            else:
                return "Content not found"
                
        except Exception as e:
            return f"Error fetching article content: {str(e)}"

# Example configuration for a few news sources
news_sources = [
    {
        'name': 'CNN',
        'url': 'https://www.cnn.com',
        'article_selector': '.container_lead-plus-headlines__item',
        'title_selector': '.container__headline-text',
        'link_selector': 'a',
        'content_selector': '.article__content',
        'date_selector': '.timestamp'
    },
    {
        'name': 'BBC',
        'url': 'https://www.bbc.com/news',
        'article_selector': '.gs-c-promo',
        'title_selector': '.gs-c-promo-heading__title',
        'link_selector': '.gs-c-promo-heading',
        'content_selector': '.article__body-content',
        'date_selector': '.date'
    },
    {
        'name': 'Reuters',
        'url': 'https://www.reuters.com',
        'article_selector': '.story-card',
        'title_selector': '.story-card__heading__text',
        'link_selector': 'a.story-card__heading__text',
        'content_selector': '.article-body__content',
        'date_selector': '.story-card__published-date'
    }
]

# For demonstration, you can run this directly
if __name__ == "__main__":
    collector = NewsCollector(news_sources)
    articles = collector.scrape_all_sources()
    print(f"Collected {len(articles)} articles in total")