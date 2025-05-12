# RAG-based News Chatbot

A web-accessible chatbot that aggregates news from various sources and allows users to query it using natural language. The system uses Retrieval-Augmented Generation (RAG) with the Llama language model to provide informed responses about current events.

## Features

- **News Aggregation**: Scrapes news from multiple websites and sources
- **RAG Architecture**: Uses vector search to retrieve relevant articles for answering queries
- **Llama Integration**: Leverages the open-source Llama model for natural language generation
- **Web Interface**: Simple, responsive UI for querying the system
- **Automatic Updates**: Scheduled news fetching to keep information current
- **Source Attribution**: Cites sources for all information provided

## Architecture

The system consists of three main components:

1. **News Collector**: Scrapes news articles from configured sources and stores them in JSON format
2. **RAG System**: Indexes articles using embeddings and uses Llama for query processing
3. **Web Interface**: Flask-based web application for user interaction

## Setup and Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- 8GB+ RAM (recommended for running Llama models)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-rag-chatbot.git
   cd news-rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface at: `http://localhost:5000`

### Docker Deployment

For easier deployment, use Docker Compose:

1. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

2. Access the web interface at: `http://localhost:5000`

3. View logs:
   ```bash
   docker-compose logs -f
   ```

## Configuration

### News Sources

You can configure news sources in the `news_scraper.py` file. Each source needs:

- `name`: Identifier for the source
- `url`: The URL to scrape
- `article_selector`: CSS selector for article elements
- `title_selector`: CSS selector for article titles
- `link_selector`: CSS selector for article links
- `content_selector`: CSS selector for article content
- `date_selector`: CSS selector for article dates

Example:
```python
{
    'name': 'BBC',
    'url': 'https://www.bbc.com/news',
    'article_selector': '.gs-c-promo',
    'title_selector': '.gs-c-promo-heading__title',
    'link_selector': '.gs-c-promo-heading',
    'content_selector': '.article__body-content',
    'date_selector': '.date'
}
```

### Llama Model

By default, the system uses `meta-llama/Llama-2-7b-chat-hf` from Hugging Face. You can change the model in `rag_system.py`:

```python
rag_system = NewsRAGSystem(
    model_name="your-preferred-llama-model",
    embedding_model="your-preferred-embedding-model",
    device="cuda"  # Use "cpu" for CPU-only systems
)
```

## Usage

### Web Interface

1. Open the web interface in a browser at `http://localhost:5000`
2. Enter your news-related questions in the input field
3. The system will respond with relevant information and cite sources
4. Click "Refresh News Data" to manually trigger a news update

### Query Examples

- "What's the latest news about climate change?"
- "Tell me about recent developments in artificial intelligence"
- "What happened in the stock market today?"
- "Summarize the current situation in [region/country]"
- "What are the major technology announcements this week?"

## Customization

### Adding New Sources

To add new news sources, modify the `news_sources` list in `news_scraper.py`.

### Changing Update Frequency

To change how often news is refreshed, modify the scheduler in `app.py`:

```python
# Default is every 6 hours
schedule.every(6).hours.do(scrape_and_update_index)

# Change to every 12 hours
schedule.every(12).hours.do(scrape_and_update_index)

# Or daily at midnight
schedule.every().day.at("00:00").do(scrape_and_update_index)
```

### Using Different Embedding Models

You can change the embedding model in `rag_system.py` to any compatible model from Hugging Face:

```python
self.embed_model = HuggingFaceEmbedding(
    model_name="your-preferred-embedding-model",
    device=device
)
```

## Limitations

- The system can only provide information about news it has indexed
- The quality of responses depends on the quality of the sources and the model
- Free Llama models have context limitations compared to commercial models
- Web scraping might break if news sites change their structure

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
