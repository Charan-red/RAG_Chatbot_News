from flask import Flask, request, jsonify, render_template
import threading
import schedule
import time
from datetime import datetime
import os
import logging

from news_scraper import NewsCollector, news_sources
from rag_system import NewsRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the RAG system
rag_system = NewsRAGSystem()

# Initialize news collector
news_collector = NewsCollector(news_sources)

def scrape_and_update_index():
    """Scrape news and update the index"""
    try:
        logger.info("Starting scheduled news scraping")
        articles = news_collector.scrape_all_sources()
        if articles:
            logger.info(f"Scraped {len(articles)} articles, updating index")
            rag_system.add_articles(articles)
        else:
            logger.warning("No articles scraped")
    except Exception as e:
        logger.error(f"Error in scheduled scraping: {str(e)}")

# Function to run scheduled tasks
def run_scheduler():
    # Schedule scraping every 6 hours
    schedule.every(6).hours.do(scrape_and_update_index)
    
    # Initial scraping if no data exists
    if not os.path.exists("./data") or not os.listdir("./data"):
        logger.info("No data found, performing initial scraping")
        scrape_and_update_index()
    else:
        # Load existing data
        logger.info("Loading existing news articles")
        rag_system.load_from_json_files()
    
    # Run scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle queries to the RAG system"""
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Received query: {user_query}")
        
        # Process query through RAG system
        result = rag_system.query(user_query)
        
        # Add timestamp to response
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': 'Error processing your query'}), 500

@app.route('/refresh', methods=['POST'])
def refresh():
    """Force refresh of news data"""
    try:
        # Start a new thread to avoid blocking the response
        threading.Thread(target=scrape_and_update_index).start()
        return jsonify({'message': 'News refresh started'})
    except Exception as e:
        logger.error(f"Error starting refresh: {str(e)}")
        return jsonify({'error': 'Error starting refresh'}), 500

if __name__ == '__main__':
    # Make sure the templates and static folders exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)