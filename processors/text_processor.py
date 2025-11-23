import requests
import json
import re
from datetime import datetime, timedelta
import urllib.parse
import logging
import sys
import asyncio
import uuid
import schedule
import time
import threading
from typing import Dict, List, Optional, Any
import yfinance as yf
import sqlite3
from dataclasses import dataclass
import base64
import io
from PIL import Image
import os

# Set up enhanced logging with file output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

@dataclass
class MonitoringTask:
    id: str
    name: str
    type: str
    parameters: Dict[str, Any]
    schedule_pattern: str
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    alerts_sent: int = 0

class EnhancedFreeChatBot:
    def __init__(self):
        self.conversation_history = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/555.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/555.36'
        })
        
        # API Keys
        self.openweathermap_api_key = "YOUR_WEATHER_API"
        self.google_generative_ai_api_key = "YOUR_GEMINI_API"
        self.newsapi_org_api_key = "YOUR_NEWS_API"
        
        # Financial APIs
        self.alpha_vantage_api_key = "YOUR_ALPHA_VANTAGE_KEY"
        self.finnhub_api_key = "YOUR_FINNHUB_KEY"
        
        # Enhanced features
        self.monitoring_tasks = {}
        self.user_preferences = {}
        self.financial_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Initialize database
        self._initialize_database()
        
        # Start background services
        self._start_background_services()
        
        # Validate API keys
        self._validate_api_keys()

    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect('chatbot_data.db', check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    parameters TEXT,
                    schedule_pattern TEXT,
                    is_active BOOLEAN,
                    created_at TIMESTAMP,
                    last_run TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS financial_watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    target_price REAL,
                    alert_type TEXT,
                    created_at TIMESTAMP,
                    is_active BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    role TEXT,
                    content TEXT,
                    intent TEXT,
                    confidence_score REAL,
                    execution_time REAL,
                    features_used TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")

    def _validate_api_keys(self):
        """Validate and log API key status"""
        api_status = {
            "Google Generative AI": bool(self.google_generative_ai_api_key and self.google_generative_ai_api_key != "YOUR_KEY"),
            "OpenWeatherMap": bool(self.openweathermap_api_key and self.openweathermap_api_key != "YOUR_KEY"),
            "NewsAPI": bool(self.newsapi_org_api_key and self.newsapi_org_api_key != "YOUR_KEY"),
            "Alpha Vantage": bool(self.alpha_vantage_api_key and self.alpha_vantage_api_key != "YOUR_ALPHA_VANTAGE_KEY"),
            "Finnhub": bool(self.finnhub_api_key and self.finnhub_api_key != "YOUR_FINNHUB_KEY")
        }
        
        for service, status in api_status.items():
            if status:
                print(f"✅ {service} API configured")
            else:
                print(f"⚠️ {service} API not configured - features will be limited")

    def _start_background_services(self):
        """Start background monitoring services"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logging.info("Background scheduler started")

    async def _call_gemini_text_model(self, prompt, chat_history=None):
        """Enhanced Gemini API call - keeps last 10 interactions"""
        if chat_history is None:
            chat_history = []

        if not self.google_generative_ai_api_key or self.google_generative_ai_api_key == "YOUR_KEY":
            return "Gemini AI is not configured. Please set your Google Cloud API Key."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                formatted_history = []
                for entry in chat_history[-10:]:
                    role = 'user' if entry['role'] == 'user' else 'model'
                    formatted_history.append({"role": role, "parts": [{"text": entry['content'][:300]}]})

                formatted_history.append({"role": "user", "parts": [{"text": prompt}]})

                payload = {
                    "contents": formatted_history,
                    "generationConfig": {
                        "temperature": 0.3,
                        "topK": 20,
                        "topP": 0.8,
                        "maxOutputTokens": 1024
                    }
                }
                
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.google_generative_ai_api_key}"
                
                response = self.session.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=30)
                response.raise_for_status()

                result = response.json()

                if result.get("candidates") and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if candidate.get("content") and candidate["content"].get("parts"):
                        return candidate["content"]["parts"][0].get("text", "No response generated.")
                
                logging.error(f"Unexpected Gemini response structure: {result}")
                return "I encountered an issue generating a response. Please try again."
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return "I'm having trouble connecting to my AI brain. Please check your internet connection and try again."
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logging.error(f"Unexpected error in Gemini call: {e}")
                return "An unexpected error occurred. Please try again."

    def _extract_city(self, query):
        """FIXED: Universal city extraction for ALL cities worldwide"""
        query_clean = query.lower().strip()
        
        logging.info(f"🔍 Extracting city from: '{query_clean}'")
        
        # FIXED: Comprehensive patterns for ANY city worldwide - improved to handle "weather in Lahore"
        patterns = [
            r'weather\s+in\s+([A-Za-z\s\-\']+?)(?:\s+tonight|\s+tomorrow|\s+today|\s+for|\s+next|\s+please|\s+focus|\s+only|\s+on|\s+answering|\s+the|\s+current|\s+question|\s*$)',
            r'weather\s+for\s+([A-Za-z\s\-\']+?)(?:\s+tonight|\s+tomorrow|\s+today|\s+for|\s+next|\s+please|\s+focus|\s+only|\s+on|\s+answering|\s+the|\s+current|\s+question|\s*$)',
            r'weather\s+at\s+([A-Za-z\s\-\']+?)(?:\s+tonight|\s+tomorrow|\s+today|\s+for|\s+next|\s+please|\s+focus|\s+only|\s+on|\s+answering|\s+the|\s+current|\s+question|\s*$)',
            r'weather\s+of\s+([A-Za-z\s\-\']+?)(?:\s+tonight|\s+tomorrow|\s+today|\s+for|\s+next|\s+please|\s+focus|\s+only|\s+on|\s+answering|\s+the|\s+current|\s+question|\s*$)',
            r'^([A-Za-z\s\-\']+?)\s+weather(?:\s+tonight|\s+tomorrow|\s+today|\s+for|\s+next|\s+please|\s+focus|\s+only|\s+on|\s+answering|\s+the|\s+current|\s+question|\s*$)',
            r'(?:temperature|forecast)\s+(?:in|for|at|of)\s+([A-Za-z\s\-\']+?)(?:\s|$)',
            r'how(?:\'s| is)\s+(?:the\s+)?weather\s+(?:in|at|for)\s+([A-Za-z\s\-\']+)',
            r'(?:tonight|tomorrow|today)\s+(?:in|at|for)\s+([A-Za-z\s\-\']+)',
            r'next\s+\w+\s+days?\s+(?:in|at|for)\s+([A-Za-z\s\-\']+)',
            r'weather\s+([A-Za-z\s\-\']{2,})(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_clean, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                
                # Clean up unwanted phrases more aggressively
                stop_words = ['focus', 'only', 'on', 'answering', 'the', 'current', 'question', 
                             'please', 'sir', 'tonight', 'tomorrow', 'today', 'for', 'next', 'without', 'referencing', 'previous', 'conversations']
                
                words = city.split()
                # Remove stop words from the end
                while words and words[-1] in stop_words:
                    words.pop()
                # Remove stop words from the beginning
                while words and words[0] in stop_words:
                    words.pop(0)
                
                city = ' '.join(words)
                city = city.strip(' ,.-')
                
                # Validation: 2+ chars, no numbers, not common words
                if city and len(city) >= 2:
                    if not any(char.isdigit() for char in city):
                        non_cities = ['and', 'the', 'or', 'but', 'with', 'from', 'about', 'in', 'at', 'for', 'of']
                        if city.lower() not in non_cities:
                            logging.info(f"✅ City extracted: '{city}'")
                            return city
        
        # Fallback: try to extract any word that looks like a city name
        words = query_clean.split()
        for i, word in enumerate(words):
            if word == 'in' and i + 1 < len(words):
                potential_city = words[i + 1]
                if len(potential_city) >= 2 and not any(char.isdigit() for char in potential_city):
                    potential_city = potential_city.strip(' ,.-')
                    non_cities = ['and', 'the', 'or', 'but', 'with', 'from', 'about', 'in', 'at', 'for', 'of', 'focus', 'only', 'on', 'answering', 'the', 'current', 'question']
                    if potential_city.lower() not in non_cities:
                        logging.info(f"✅ City extracted (fallback): '{potential_city}'")
                        return potential_city
        
        logging.warning(f"⚠️ No city found in query: '{query_clean}'")
        return None

    async def _general_chat(self, query):
        """Enhanced general chat with better context"""
        if query.lower() in ['status', 'stats', 'system']:
            return self.get_system_stats()
        elif 'capabilities' in query.lower() or 'what can you do' in query.lower():
            return self.get_help()
        
        return await self._call_gemini_text_model(query, self.conversation_history)

    async def _generate_image_from_text(self, prompt):
        """Direct image generation"""
        try:
            enhanced_prompt = self._enhance_image_prompt(prompt)
            unique_prompt = f"{enhanced_prompt} --seed={uuid.uuid4().hex[:8]}"
            encoded_prompt = urllib.parse.quote_plus(unique_prompt)
            
            image_services = [
                f"https://image.pollinations.ai/prompt/{encoded_prompt}",
                f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model=flux",
                f"https://image.pollinations.ai/prompt/{encoded_prompt}?enhance=true"
            ]
            
            print("🎨 Generating image via pollinations.ai...")
            
            for service_url in image_services:
                try:
                    response = self.session.get(service_url, timeout=120)
                    response.raise_for_status()
                    
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    
                    image_filename = f"generated_image_{uuid.uuid4().hex[:8]}.png"
                    image_path = os.path.join("static", "images", image_filename)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    
                    return {
                        'type': 'image_generation',
                        'status': 'success',
                        'message': f"✅ Image generated successfully!",
                        'image_url': service_url,
                        'local_path': image_path,
                        'image_base64': image_base64,
                        'prompt_used': enhanced_prompt
                    }
                    
                except requests.exceptions.RequestException:
                    continue
            
            return {
                'type': 'image_generation',
                'status': 'error',
                'message': "❌ Image generation failed. All services busy. Please try again."
            }
            
        except Exception as e:
            logging.error(f"Image generation error: {e}")
            return {
                'type': 'image_generation',
                'status': 'error',
                'message': f"❌ Image generation error: {str(e)}"
            }

    def _enhance_image_prompt(self, prompt):
        """Enhance image generation prompts"""
        enhancements = {
            'style': ', highly detailed, professional quality, vibrant colors',
            'technical': ', 8k resolution, sharp focus, perfect lighting',
            'artistic': ', artistic masterpiece, trending on artstation'
        }
        
        if any(word in prompt.lower() for word in ['realistic', 'photo', 'portrait']):
            return prompt + enhancements['technical']
        elif any(word in prompt.lower() for word in ['art', 'painting', 'drawing', 'creative']):
            return prompt + enhancements['artistic']
        else:
            return prompt + enhancements['style']

    async def _get_stock_data(self, symbol):
        """Get real-time stock data using multiple sources"""
        symbol = symbol.upper()
        
        cache_key = f"stock_{symbol}"
        if cache_key in self.financial_cache:
            cached_data, timestamp = self.financial_cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                previous_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
                
                stock_data = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'current_price': round(current_price, 2),
                    'previous_close': round(previous_close, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_percent, 2),
                    'volume': info.get('volume', 'N/A'),
                    'market_cap': info.get('marketCap', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'day_high': round(hist['High'].iloc[-1], 2) if not hist.empty else 'N/A',
                    'day_low': round(hist['Low'].iloc[-1], 2) if not hist.empty else 'N/A',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.financial_cache[cache_key] = (stock_data, datetime.now())
                return stock_data
                
        except Exception as e:
            logging.error(f"Error fetching stock data for {symbol}: {e}")
        
        if self.alpha_vantage_api_key and self.alpha_vantage_api_key != "YOUR_ALPHA_VANTAGE_KEY":
            try:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_api_key}"
                response = self.session.get(url, timeout=10)
                data = response.json()
                
                quote = data.get('Global Quote', {})
                if quote:
                    stock_data = {
                        'symbol': symbol,
                        'current_price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': float(quote.get('10. change percent', '0%').rstrip('%')),
                        'volume': quote.get('06. volume', 'N/A'),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    self.financial_cache[cache_key] = (stock_data, datetime.now())
                    return stock_data
                    
            except Exception as e:
                logging.error(f"Alpha Vantage API error: {e}")
        
        return None

    async def _format_financial_response(self, stock_data):
        """Format financial data into user-friendly response"""
        if not stock_data:
            return "❌ Unable to fetch stock data. Please check the symbol and try again."
        
        symbol = stock_data['symbol']
        price = stock_data['current_price']
        change = stock_data['change']
        change_percent = stock_data['change_percent']
        
        trend_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        color_indicator = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
        
        response = f"{trend_emoji} **{symbol}** Stock Information {color_indicator}\n\n"
        
        if 'company_name' in stock_data:
            response += f"**Company:** {stock_data['company_name']}\n"
        
        response += f"**Current Price:** ${price}\n"
        response += f"**Change:** ${change:+.2f} ({change_percent:+.2f}%)\n"
        
        if stock_data.get('day_high') != 'N/A':
            response += f"**Day Range:** ${stock_data['day_low']} - ${stock_data['day_high']}\n"
        
        if stock_data.get('volume') != 'N/A':
            response += f"**Volume:** {stock_data['volume']:,}\n"
        
        if stock_data.get('market_cap') != 'N/A':
            market_cap = stock_data['market_cap']
            if isinstance(market_cap, (int, float)):
                market_cap_formatted = f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.2f}M"
                response += f"**Market Cap:** {market_cap_formatted}\n"
        
        if stock_data.get('pe_ratio') != 'N/A':
            response += f"**P/E Ratio:** {stock_data['pe_ratio']:.2f}\n"
        
        response += f"\n*Last updated: {stock_data['timestamp']}*"
        
        return response

    async def _create_financial_portfolio_summary(self, symbols):
        """Create portfolio summary"""
        portfolio_data = []
        total_value_change = 0
        
        for symbol in symbols:
            stock_data = await self._get_stock_data(symbol)
            if stock_data:
                portfolio_data.append(stock_data)
                total_value_change += stock_data['change_percent']
        
        if not portfolio_data:
            return "❌ Unable to fetch portfolio data."
        
        avg_change = total_value_change / len(portfolio_data)
        portfolio_trend = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
        
        response = f"{portfolio_trend} **Portfolio Summary** ({len(portfolio_data)} stocks)\n\n"
        response += f"**Average Change:** {avg_change:+.2f}%\n\n"
        
        for stock in portfolio_data:
            trend = "🟢" if stock['change'] > 0 else "🔴" if stock['change'] < 0 else "🟡"
            response += f"{trend} **{stock['symbol']}:** ${stock['current_price']} ({stock['change_percent']:+.2f}%)\n"
        
        response += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        return response

    async def _create_monitoring_task(self, task_type, parameters, schedule_pattern="daily"):
        """Create monitoring task"""
        task_id = str(uuid.uuid4())[:8]
        
        task = MonitoringTask(
            id=task_id,
            name=parameters.get('name', f'{task_type}_{task_id}'),
            type=task_type,
            parameters=parameters,
            schedule_pattern=schedule_pattern
        )
        
        self.monitoring_tasks[task_id] = task
        self._schedule_monitoring_task(task)
        self._save_monitoring_task_to_db(task)
        
        return task_id

    def _schedule_monitoring_task(self, task):
        """Schedule monitoring task"""
        def execute_task():
            try:
                asyncio.create_task(self._execute_monitoring_task(task))
            except Exception as e:
                logging.error(f"Monitoring task execution error: {e}")
        
        if task.schedule_pattern == "daily":
            schedule.every().day.at("09:00").do(execute_task)
        elif task.schedule_pattern == "hourly":
            schedule.every().hour.do(execute_task)
        elif task.schedule_pattern.startswith("every_"):
            interval = int(task.schedule_pattern.split('_')[1].replace('min', ''))
            schedule.every(interval).minutes.do(execute_task)

    async def _execute_monitoring_task(self, task):
        """Execute monitoring task"""
        try:
            task.last_run = datetime.now()
            alert_sent = False
            
            if task.type == "stock_alert":
                alert_sent = await self._check_stock_alerts(task)
            elif task.type == "news_monitor":
                alert_sent = await self._check_news_keywords(task)
            elif task.type == "weather_alert":
                alert_sent = await self._check_weather_conditions(task)
            
            if alert_sent:
                task.alerts_sent += 1
                self._update_monitoring_task_in_db(task)
                
        except Exception as e:
            logging.error(f"Error executing monitoring task {task.id}: {e}")

    async def _check_stock_alerts(self, task):
        """Check stock price alerts"""
        try:
            symbol = task.parameters.get('symbol')
            target_price = task.parameters.get('target_price')
            alert_type = task.parameters.get('alert_type', 'above')
            
            stock_data = await self._get_stock_data(symbol)
            if not stock_data:
                return False
            
            current_price = stock_data['current_price']
            
            should_alert = False
            if alert_type == 'above' and current_price >= target_price:
                should_alert = True
            elif alert_type == 'below' and current_price <= target_price:
                should_alert = True
            
            if should_alert:
                alert_message = f"🚨 **Stock Alert: {symbol}**\n\n"
                alert_message += f"Price has reached **${current_price}** (target: ${target_price})\n"
                alert_message += f"Change: {stock_data['change_percent']:+.2f}%\n"
                alert_message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': alert_message,
                    'timestamp': datetime.now(),
                    'type': 'stock_alert'
                })
                
                logging.info(f"Stock alert triggered for {symbol}: {current_price}")
                return True
                
        except Exception as e:
            logging.error(f"Stock alert check error: {e}")
        
        return False

    def _save_monitoring_task_to_db(self, task):
        """Save monitoring task to database"""
        try:
            conn = sqlite3.connect('chatbot_data.db', check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO monitoring_tasks 
                (id, name, type, parameters, schedule_pattern, is_active, created_at, last_run)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id, task.name, task.type, 
                json.dumps(task.parameters), task.schedule_pattern, 
                task.is_active, datetime.now(), task.last_run
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving monitoring task: {e}")

    def _update_monitoring_task_in_db(self, task):
        """Update monitoring task in database"""
        try:
            conn = sqlite3.connect('chatbot_data.db', check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE monitoring_tasks 
                SET last_run = ?, is_active = ?
                WHERE id = ?
            ''', (task.last_run, task.is_active, task.id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error updating monitoring task: {e}")

    def _detect_intent(self, text):
        """Enhanced intent detection"""
        text_lower = text.lower()

        if text_lower.startswith("ask:"): return 'general_chat_direct'
        if text_lower in ["help", "commands"]: return 'help'
        if text_lower in ["clear", "reset"]: return 'clear_history'
        if text_lower.startswith("monitor"): return 'monitoring'
        if text_lower.startswith("alert"): return 'alert_setup'

        if self._is_direct_math_query(text_lower):
            return 'math'

        financial_keywords = ['stock price', 'share price', 'portfolio', 'investment', 
                             'market cap', 'trading', 'nasdaq', 'nyse', 'dow jones', 's&p 500']
        
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_symbols = re.findall(stock_pattern, text)
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 
                       'HAS', 'WHO', 'WILL', 'WITH', 'FROM', 'THEY', 'BEEN', 'HAVE',
                       'THIS', 'THAT', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHILE',
                       'ABOUT', 'AFTER', 'COULD', 'WOULD', 'SHOULD', 'THEIR', 'THERE',
                       'THESE', 'THOSE', 'THROUGH', 'DURING', 'BEFORE', 'BETWEEN',
                       'UNDER', 'ABOVE', 'BELOW', 'ACROSS', 'ALONG', 'AROUND'}
        
        actual_symbols = [s for s in potential_symbols if s not in common_words and len(s) <= 5]
        
        if any(keyword in text_lower for keyword in financial_keywords):
            return 'financial'
        elif actual_symbols and ('price' in text_lower or 'stock' in text_lower or 'share' in text_lower):
            return 'financial'

        if self._is_multi_modal_request(text_lower):
            return 'multi_modal'

        image_keywords = [
            'create an image', 'create image', 'generate an image', 'generate image',
            'draw me', 'draw an', 'draw a', 'make an image', 'make image',
            'show me an image', 'show me image', 'picture of', 'image of',
            'generate a picture', 'generate picture', 'paint me', 'sketch',
            'artwork of', 'illustration of', 'visualize', 'create visual',
            'make picture', 'show picture', 'draw picture'
        ]
        if any(phrase in text_lower for phrase in image_keywords):
            return 'image_generation'

        if "translate" in text_lower or re.search(r"how do you say", text_lower):
            return 'translate'

        explicit_news_patterns = [
            'latest news', 'breaking news', 'news headlines', 'current news', 
            'news today', 'recent news', 'news update', 'headlines today',
            'show me news', 'get news', 'news please', 'news about'
        ]
        if any(pattern in text_lower for pattern in explicit_news_patterns):
            return 'news'

        weather_keywords = ['weather in', 'weather for', 'weather at', 'temperature in', 
                           'forecast for', 'weather tonight', 'weather tomorrow', 
                           'tonight in', 'tomorrow in', 'next three days', 'next 7 days']
        knowledge_phrases = ['tell me about', 'who was', 'what is', 'explain', 'define', 'history of']

        is_weather_query = any(keyword in text_lower for keyword in weather_keywords)
        is_knowledge_query = any(phrase in text_lower for phrase in knowledge_phrases)

        if is_weather_query and not is_knowledge_query:
            return 'weather'

        programming_keywords = ['code', 'python', 'javascript', 'debug', 'error', 'programming', 
                               'algorithm', 'function', 'class', 'loop', 'variable', 'syntax']
        if any(word in text_lower for word in programming_keywords):
            return 'programming'

        if is_knowledge_query or any(text_lower.startswith(q) for q in ['what is', 'who is', 'when was']):
            return 'knowledge'

        return 'general'

    def _is_direct_math_query(self, text_lower):
        """Detect direct math queries"""
        math_patterns = [
            r'\d+\s*[\+\-\*\/\^]\s*\d+',
            r'calculate\s+\d+',
            r'what is\s+\d+.*[\+\-\*\/].*\d+',
            r'solve\s+\d+',
            r'\d+\s*\%\s*of\s*\d+',
            r'\d+\s*power\s*\d+',
            r'\d+\s*to\s*the\s*power\s*of\s*\d+',
            r'square\s*root\s*of\s*\d+',
            r'\d+\s*squared',
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text_lower):
                return True
        
        if any(word in text_lower for word in ['calculate', 'compute', 'solve']) and re.search(r'\d', text_lower):
            return True
            
        return False

    def _is_multi_modal_request(self, text_lower):
        """Better multi-modal detection"""
        
        if 'after giving your reasoning' in text_lower or 'after your reasoning' in text_lower:
            if any(word in text_lower for word in ['create', 'generate', 'image', 'picture', 'visual', 'show']):
                return True
        
        if 'after that' in text_lower:
            if any(word in text_lower for word in ['create', 'generate', 'image', 'picture', 'draw', 'show']):
                return True
        
        if 'now, after' in text_lower or 'now after' in text_lower:
            if any(word in text_lower for word in ['create', 'generate', 'image', 'picture', 'visual']):
                return True
        
        multi_modal_patterns = [
            ('tell me about', 'create image'),
            ('explain', 'generate image'),
            ('describe', 'show me image'),
            ('what is', 'create image'),
            ('define', 'illustrate'),
            ('explain', 'draw'),
            ('tell me about', 'generate picture'),
        ]
        
        for text_pattern, image_pattern in multi_modal_patterns:
            if text_pattern in text_lower and image_pattern in text_lower:
                return True
        
        if ' and ' in text_lower:
            has_explanation = any(word in text_lower for word in ['explain', 'tell', 'describe', 'what', 'how'])
            has_image = any(phrase in text_lower for phrase in ['create image', 'generate image', 'show image', 'draw', 'picture'])
            if has_explanation and has_image:
                return True
            
        return False

    async def process_message(self, user_input):
        """Enhanced message processing"""
        start_time = datetime.now()
        print("\n🤔 Thinking...")
        
        self.conversation_history.append({
            'role': 'user', 
            'content': user_input, 
            'timestamp': start_time
        })
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        intent = self._detect_intent(user_input)
        confidence_score = 0.8
        features_used = [intent]
        
        try:
            if intent == 'multi_modal':
                response_text = await self._handle_multi_modal_request(user_input)
            elif intent == 'financial':
                response_text = await self._handle_financial_queries(user_input)
            elif intent == 'image_generation':
                response_data = await self._handle_image_generation(user_input)
                response_text = response_data
                features_used.append('image_generation')
            elif intent == 'monitoring':
                response_text = await self._handle_monitoring_setup(user_input)
            elif intent == 'alert_setup':
                response_text = await self._handle_alert_setup(user_input)
            elif intent == 'news':
                response_text = await self._get_news(user_input)
            elif intent == 'weather':
                response_text = await self._get_weather(user_input)
            elif intent == 'programming':
                response_text = await self._get_programming_help(user_input)
            elif intent == 'math':
                response_text = await self._solve_math_with_gemini(user_input)
            elif intent == 'translate':
                response_text = await self._translate_text(user_input)
            elif intent == 'knowledge':
                response_text = await self._get_knowledge(user_input)
            elif intent == 'general_chat_direct':
                response_text = await self._general_chat(user_input[4:].strip())
            else:
                response_text = await self._general_chat(user_input)
                
        except Exception as e:
            logging.error(f"Error processing intent '{intent}': {e}")
            response_text = self._fallback_response(user_input)
            confidence_score = 0.3

        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.conversation_history.append({
            'role': 'assistant', 
            'content': response_text if isinstance(response_text, str) else str(response_text), 
            'timestamp': datetime.now(),
            'intent': intent,
            'confidence_score': confidence_score,
            'execution_time': execution_time
        })
        
        self._log_conversation_to_db(user_input, response_text, intent, confidence_score, execution_time, features_used)
        
        return response_text

    async def _handle_multi_modal_request(self, user_input):
        """Handle multi-modal requests"""
        try:
            text_topic, image_prompt = self._parse_multi_modal_request(user_input)
            
            print("🔄 Processing multi-modal request (text + image)...")
            
            text_response = await self._call_gemini_text_model(
                f"""Provide a comprehensive but well-structured explanation about: {text_topic}

Format your response as clear, flowing paragraphs without excessive bullet points. Focus on key insights and transformative impacts in a narrative style.""",
                self.conversation_history[-10:]
            )
            
            image_result = await self._generate_image_from_text(image_prompt)
            
            if isinstance(image_result, dict) and image_result.get('status') == 'success':
                combined_response = text_response.strip()
                combined_response += f"\n\n🖼️ **View the generated image:** {image_result['image_url']}\n"
                combined_response += f"*Prompt: {image_result['prompt_used'][:80]}...*"
                
                return {
                    'type': 'multi_modal',
                    'text_response': text_response,
                    'image_data': image_result,
                    'combined_response': combined_response,
                    'unified_output': True
                }
            else:
                return f"{text_response}\n\n❌ Image generation issue: {image_result.get('message', 'Service temporarily unavailable')}"
                
        except Exception as e:
            logging.error(f"Multi-modal processing error: {e}")
            return f"I've processed your question but encountered an issue with image generation: {str(e)}"

    def _parse_multi_modal_request(self, user_input):
        """Better parsing for multi-modal requests"""
        text_lower = user_input.lower()
        
        if 'after giving your reasoning' in text_lower or 'after that' in text_lower:
            if 'after giving your reasoning' in text_lower:
                parts = user_input.split('after giving your reasoning', 1)
            else:
                parts = user_input.split('after that', 1)
            
            text_topic = parts[0].strip()
            image_part = parts[1].strip() if len(parts) > 1 else text_topic
            
            image_part = re.sub(r'^[,\s]*(create|generate|draw|make|show)\s*(a|an)?\s*(detailed)?\s*(image|picture|visualization|visual)?\s*(that|of|showing|representing)?\s*', '', image_part, flags=re.IGNORECASE)
            image_prompt = image_part.strip() if image_part else text_topic
        
        elif 'now, after' in text_lower or 'now after' in text_lower:
            parts = re.split(r'now,?\s*after', user_input, flags=re.IGNORECASE)
            text_topic = parts[0].strip()
            image_part = parts[1].strip() if len(parts) > 1 else text_topic
            image_part = re.sub(r'^[,\s]*(giving your reasoning|that|this)[,\s]*(create|generate|draw)\s*(a|an)?\s*(detailed)?\s*(image|picture)?\s*(that|of)?\s*', '', image_part, flags=re.IGNORECASE)
            image_prompt = image_part.strip() if image_part else text_topic
        
        elif 'and create image' in text_lower or 'and generate image' in text_lower:
            if 'and create image' in text_lower:
                parts = user_input.split('and create image', 1)
            else:
                parts = user_input.split('and generate image', 1)
            
            text_topic = parts[0].strip()
            image_prompt = parts[1].strip() if len(parts) > 1 else text_topic
        
        else:
            text_topic = user_input
            image_prompt = user_input.replace('tell me about', '').replace('explain', '').replace('what is', '').strip()
        
        return text_topic, image_prompt

    async def _solve_math_with_gemini(self, query):
        """Solve math problems using Gemini"""
        try:
            math_expr = self._extract_math_expression(query)
            if math_expr:
                try:
                    expr = math_expr.replace('^', '**').replace('power', '**')
                    
                    if all(c in '0123456789.+-*/() ' for c in expr):
                        result = eval(expr)
                        return f"🧮 **Math Solution**\n\n**Expression:** {math_expr}\n**Result:** {result}"
                except:
                    pass
            
            math_prompt = f"""You are a mathematics expert. Solve this math problem step by step:

Problem: {query}

Please provide:
1. Step-by-step solution
2. Final answer clearly marked
3. Brief explanation of the method used

Keep your response focused on the mathematical solution."""
            
            gemini_response = await self._call_gemini_text_model(math_prompt, [])
            
            return f"🧮 **Math Solution**\n\n{gemini_response}"
            
        except Exception as e:
            logging.error(f"Math solving error: {e}")
            return f"❌ Error solving math problem: {str(e)}. Please check your expression and try again."

    def _extract_math_expression(self, query):
        """Extract mathematical expression"""
        patterns = [
            r'(\d+\s*[\+\-\*\/\^]\s*\d+(?:\s*[\+\-\*\/\^]\s*\d+)*)',
            r'(\d+\s*power\s*\d+)',
            r'(\d+\s*to\s*the\s*power\s*of\s*\d+)',
            r'(\d+\s*\%\s*of\s*\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    async def _handle_financial_queries(self, query):
        """Handle financial queries"""
        query_lower = query.lower()
        
        stock_pattern = r'\b([A-Z]{2,5})\b'
        potential_symbols = re.findall(stock_pattern, query)
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAS', 
                       'WHO', 'WILL', 'WITH', 'FROM', 'THEY', 'BEEN', 'HAVE', 'THIS', 'THAT',
                       'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHILE', 'ABOUT', 'AFTER', 'COULD',
                       'WOULD', 'SHOULD', 'THEIR', 'THERE', 'THESE', 'THOSE'}
        symbols = [s for s in potential_symbols if s not in common_words and len(s) <= 5]
        
        if 'portfolio' in query_lower and len(symbols) > 1:
            return await self._create_financial_portfolio_summary(symbols)
        
        elif symbols:
            symbol = symbols[0]
            stock_data = await self._get_stock_data(symbol)
            return await self._format_financial_response(stock_data)
        
        elif any(term in query_lower for term in ['market', 'dow', 'nasdaq', 's&p', 'spy']):
            major_indices = ['SPY', 'QQQ', 'DIA']
            return await self._create_financial_portfolio_summary(major_indices)
        
        else:
            financial_prompt = f"As a financial advisor AI, please answer this question: {query}"
            return await self._call_gemini_text_model(financial_prompt, self.conversation_history)

    async def _handle_monitoring_setup(self, query):
        """Handle monitoring setup"""
        query_lower = query.lower()
        
        if 'stock' in query_lower or 'price' in query_lower:
            return await self._setup_stock_monitoring(query)
        elif 'news' in query_lower:
            return await self._setup_news_monitoring(query)
        elif 'weather' in query_lower:
            return await self._setup_weather_monitoring(query)
        else:
            return """🔍 **Monitoring Options Available:**
            
1. **Stock Monitoring**: `monitor AAPL above 150`
2. **News Monitoring**: `monitor news about AI`
3. **Weather Monitoring**: `monitor weather in London for rain`

Type `list monitors` to see active monitoring tasks."""

    async def _setup_stock_monitoring(self, query):
        """Setup stock monitoring"""
        pattern = r'monitor\s+([A-Z]{1,5})\s+(above|below)\s+(\d+(?:\.\d+)?)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            symbol = match.group(1).upper()
            alert_type = match.group(2).lower()
            target_price = float(match.group(3))
            
            stock_data = await self._get_stock_data(symbol)
            if not stock_data:
                return f"❌ Could not find stock data for {symbol}. Please check the symbol."
            
            task_id = await self._create_monitoring_task('stock_alert', {
                'name': f'{symbol} Price Alert',
                'symbol': symbol,
                'target_price': target_price,
                'alert_type': alert_type
            }, 'every_15min')
            
            return f"✅ **Stock Alert Created!**\n\nMonitoring **{symbol}** for price {alert_type} ${target_price}\nCurrent price: ${stock_data['current_price']}\nTask ID: {task_id}\n\n*You'll receive alerts when the condition is met.*"
        
        else:
            return "❌ Invalid format. Use: `monitor SYMBOL above/below PRICE`\nExample: `monitor AAPL above 150`"

    async def _setup_news_monitoring(self, query):
        """Setup news monitoring"""
        pattern = r'monitor\s+news\s+(?:about\s+)?(.+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            keywords = match.group(1).strip()
            
            task_id = await self._create_monitoring_task('news_monitor', {
                'name': f'News Alert: {keywords}',
                'keywords': keywords.split(),
                'sources': ['general']
            }, 'every_30min')
            
            return f"✅ **News Alert Created!**\n\nMonitoring news for: **{keywords}**\nTask ID: {task_id}"
        
        else:
            return "❌ Invalid format. Use: `monitor news about KEYWORDS`"

    async def _check_news_keywords(self, task):
        """Check news keywords"""
        try:
            keywords = task.parameters.get('keywords', [])
            
            news_response = await self._get_news("latest news")
            if not news_response or "error" in news_response.lower():
                return False
            
            news_lower = news_response.lower()
            found_keywords = [kw for kw in keywords if kw.lower() in news_lower]
            
            if found_keywords:
                alert_message = f"📰 **News Alert: {task.parameters['name']}**\n\n"
                alert_message += f"Keywords found: {', '.join(found_keywords)}\n\n"
                alert_message += f"Recent headlines:\n{news_response[:500]}..."
                
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': alert_message,
                    'timestamp': datetime.now(),
                    'type': 'news_alert'
                })
                
                return True
                
        except Exception as e:
            logging.error(f"News monitoring error: {e}")
        
        return False

    async def _setup_weather_monitoring(self, query):
        """Setup weather monitoring"""
        pattern = r'monitor\s+weather\s+in\s+([A-Za-z\s]+)\s+for\s+(.+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            location = match.group(1).strip()
            condition = match.group(2).strip()
            
            task_id = await self._create_monitoring_task('weather_alert', {
                'name': f'Weather Alert: {location}',
                'location': location,
                'condition': condition
            }, 'every_60min')
            
            return f"✅ **Weather Alert Created!**\n\nMonitoring weather in **{location}** for: {condition}\nTask ID: {task_id}"
        
        else:
            return "❌ Invalid format. Use: `monitor weather in LOCATION for CONDITION`"

    async def _handle_alert_setup(self, query):
        """Handle alert setup"""
        query_lower = query.lower()
        
        if 'list' in query_lower:
            return self._list_monitoring_tasks()
        elif 'stop' in query_lower or 'delete' in query_lower:
            return await self._stop_monitoring_task(query)
        else:
            return "Use `list alerts` to see active alerts or `stop alert TASK_ID` to stop monitoring."

    def _list_monitoring_tasks(self):
        """List monitoring tasks"""
        if not self.monitoring_tasks:
            return "No active monitoring tasks."
        
        response = "🔍 **Active Monitoring Tasks:**\n\n"
        
        for task_id, task in self.monitoring_tasks.items():
            if task.is_active:
                status_emoji = "🟢"
                response += f"{status_emoji} **{task.name}** (ID: {task_id})\n"
                response += f"   Type: {task.type}\n"
                response += f"   Schedule: {task.schedule_pattern}\n"
                response += f"   Alerts sent: {task.alerts_sent}\n"
                if task.last_run:
                    response += f"   Last run: {task.last_run.strftime('%Y-%m-%d %H:%M')}\n"
                response += "\n"
        
        return response

    async def _stop_monitoring_task(self, query):
        """Stop monitoring task"""
        pattern = r'(?:stop|delete)\s+(?:alert|monitor|task)\s+([a-f0-9]{8})'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            task_id = match.group(1)
            
            if task_id in self.monitoring_tasks:
                self.monitoring_tasks[task_id].is_active = False
                self._update_monitoring_task_in_db(self.monitoring_tasks[task_id])
                return f"✅ Monitoring task {task_id} has been stopped."
            else:
                return f"❌ Task {task_id} not found."
        
        else:
            return "❌ Invalid format. Use: `stop alert TASK_ID`"

    def _log_conversation_to_db(self, user_input, response, intent, confidence, execution_time, features):
        """Log conversation to database"""
        try:
            conn = sqlite3.connect('chatbot_data.db', check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO enhanced_conversations 
                (timestamp, role, content, intent, confidence_score, execution_time, features_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                'conversation',
                f"User: {user_input[:200]}... | Bot: {str(response)[:200]}...",
                intent,
                confidence,
                execution_time,
                json.dumps(features)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error logging conversation: {e}")

    async def _get_news(self, query):
        """Get news headlines"""
        if not self.newsapi_org_api_key or self.newsapi_org_api_key == "YOUR_KEY":
            return "News service not configured."
        
        try:
            country = "us"
            category = "general"
            
            if any(word in query.lower() for word in ['tech', 'technology', 'ai']):
                category = "technology"
            elif any(word in query.lower() for word in ['business', 'market', 'economy']):
                category = "business"
            elif any(word in query.lower() for word in ['health', 'medical']):
                category = "health"
            elif any(word in query.lower() for word in ['sports', 'football', 'basketball']):
                category = "sports"
            
            news_url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={self.newsapi_org_api_key}"
            response = self.session.get(news_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "ok" and data["articles"]:
                news_response = f"📰 **Latest {category.title()} News**\n\n"
                
                for i, article in enumerate(data["articles"][:5], 1):
                    title = article.get("title", "No Title")
                    source = article.get("source", {}).get("name", "Unknown")
                    description = article.get("description", "")
                    url = article.get("url", "")
                    
                    news_response += f"**{i}. {title}**\n"
                    news_response += f"*Source: {source}*\n"
                    if description:
                        news_response += f"{description[:100]}...\n"
                    if url:
                        news_response += f"[Read more]({url})\n"
                    news_response += "\n"
                
                return news_response
            else:
                return "No news headlines found."
                
        except Exception as e:
            logging.error(f"NewsAPI error: {e}")
            return "Couldn't fetch news. Please try again later."

    async def _get_weather(self, query):
        """FIXED: Enhanced weather for ALL cities worldwide with time-specific responses"""
        city = self._extract_city(query)
        
        if not city:
            logging.warning(f"No city extracted from query: {query}")
            return "❌ Please specify a city. Example: 'Weather in London' or 'Weather in any city name'"
        
        if not self.openweathermap_api_key or self.openweathermap_api_key == "YOUR_KEY":
            return "Weather service not configured."
        
        # FIXED: Detect time-specific requests
        query_lower = query.lower()
        is_tonight = 'tonight' in query_lower
        is_tomorrow = 'tomorrow' in query_lower
        is_today = 'today' in query_lower
        
        try:
            city_encoded = urllib.parse.quote(city)
            current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city_encoded}&appid={self.openweathermap_api_key}&units=metric"
            
            logging.info(f"🌍 Fetching weather for: '{city}'")
            
            current_response = self.session.get(current_url, timeout=10)
            current_data = current_response.json()
            
            if current_response.status_code == 200:
                temp = current_data["main"]["temp"]
                feels_like = current_data["main"]["feels_like"]
                condition = current_data["weather"][0]["description"].title()
                humidity = current_data["main"]["humidity"]
                pressure = current_data["main"]["pressure"]
                wind_speed = current_data["wind"]["speed"]
                wind_direction = current_data["wind"].get("deg", 0)
                visibility = current_data.get("visibility", 0) / 1000
                
                weather_id = current_data["weather"][0]["id"]
                weather_emoji = self._get_weather_emoji(weather_id)
                
                # FIXED: Time-specific responses
                if is_tonight:
                    response = f"{weather_emoji} **Weather in {city.title()} Tonight**\n\n"
                    response += f"🌡️ **Temperature:** {temp}°C (feels like {feels_like}°C)\n"
                    response += f"☁️ **Condition:** {condition}\n"
                    response += f"💧 **Humidity:** {humidity}%\n"
                    response += f"🌪️ **Wind:** {wind_speed} m/s ({self._get_wind_direction(wind_direction)})\n"
                    response += f"📊 **Pressure:** {pressure} hPa\n"
                    if visibility > 0:
                        response += f"👁️ **Visibility:** {visibility:.1f} km\n"
                    response += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
                    return response
                
                elif is_tomorrow:
                    # Get tomorrow's forecast
                    try:
                        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_encoded}&appid={self.openweathermap_api_key}&units=metric"
                        forecast_response = self.session.get(forecast_url, timeout=5)
                        
                        if forecast_response.status_code == 200:
                            forecast_data = forecast_response.json()
                            # Get tomorrow's forecast (around 24 hours from now)
                            tomorrow_forecast = None
                            for item in forecast_data["list"]:
                                forecast_time = datetime.fromtimestamp(item["dt"])
                                if forecast_time.date() == (datetime.now() + timedelta(days=1)).date():
                                    tomorrow_forecast = item
                                    break
                            
                            if tomorrow_forecast:
                                temp_tomorrow = tomorrow_forecast["main"]["temp"]
                                condition_tomorrow = tomorrow_forecast["weather"][0]["description"].title()
                                weather_id_tomorrow = tomorrow_forecast["weather"][0]["id"]
                                weather_emoji_tomorrow = self._get_weather_emoji(weather_id_tomorrow)
                                
                                response = f"{weather_emoji_tomorrow} **Weather in {city.title()} Tomorrow**\n\n"
                                response += f"🌡️ **Temperature:** {temp_tomorrow}°C\n"
                                response += f"☁️ **Condition:** {condition_tomorrow}\n"
                                response += f"📅 **Date:** {(datetime.now() + timedelta(days=1)).strftime('%A, %B %d')}\n"
                                response += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
                                return response
                    except Exception as forecast_error:
                        logging.error(f"Tomorrow forecast error: {forecast_error}")
                
                elif is_today:
                    response = f"{weather_emoji} **Weather in {city.title()} Today**\n\n"
                    response += f"🌡️ **Temperature:** {temp}°C (feels like {feels_like}°C)\n"
                    response += f"☁️ **Condition:** {condition}\n"
                    response += f"💧 **Humidity:** {humidity}%\n"
                    response += f"🌪️ **Wind:** {wind_speed} m/s ({self._get_wind_direction(wind_direction)})\n"
                    response += f"📊 **Pressure:** {pressure} hPa\n"
                    if visibility > 0:
                        response += f"👁️ **Visibility:** {visibility:.1f} km\n"
                    response += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
                    return response
                
                # Default: Current weather + 5-day forecast
                response = f"{weather_emoji} **Weather in {city.title()}**\n\n"
                response += f"🌡️ **Temperature:** {temp}°C (feels like {feels_like}°C)\n"
                response += f"☁️ **Condition:** {condition}\n"
                response += f"💧 **Humidity:** {humidity}%\n"
                response += f"🌪️ **Wind:** {wind_speed} m/s ({self._get_wind_direction(wind_direction)})\n"
                response += f"📊 **Pressure:** {pressure} hPa\n"
                
                if visibility > 0:
                    response += f"👁️ **Visibility:** {visibility:.1f} km\n"
                
                try:
                    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_encoded}&appid={self.openweathermap_api_key}&units=metric"
                    forecast_response = self.session.get(forecast_url, timeout=5)
                    
                    if forecast_response.status_code == 200:
                        forecast_data = forecast_response.json()
                        response += f"\n📅 **5-Day Forecast:**\n"
                        
                        for i in range(0, min(40, len(forecast_data["list"])), 8):
                            forecast_item = forecast_data["list"][i]
                            date = datetime.fromtimestamp(forecast_item["dt"])
                            temp_forecast = forecast_item["main"]["temp"]
                            condition_forecast = forecast_item["weather"][0]["description"].title()
                            
                            response += f"  {date.strftime('%a %m/%d')}: {temp_forecast}°C, {condition_forecast}\n"
                
                except Exception as forecast_error:
                    logging.error(f"Forecast fetch error: {forecast_error}")
                
                response += f"\n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
                return response
            
            else:
                error_message = current_data.get("message", "Unknown error")
                logging.error(f"❌ Weather API error for '{city}': {error_message}")
                return f"❌ Couldn't fetch weather for '{city}': {error_message}\nPlease check the city name and try again."
                
        except Exception as e:
            logging.error(f"Weather API error: {e}")
            return "⚠️ Weather service temporarily unavailable."

    def _get_weather_emoji(self, weather_id):
        """Get weather emoji"""
        if 200 <= weather_id <= 232:
            return "⛈️"
        elif 300 <= weather_id <= 321:
            return "🌦️"
        elif 500 <= weather_id <= 531:
            return "🌧️"
        elif 600 <= weather_id <= 622:
            return "❄️"
        elif 701 <= weather_id <= 781:
            return "🌫️"
        elif weather_id == 800:
            return "☀️"
        elif 801 <= weather_id <= 804:
            return "⛅"
        else:
            return "🌤️"

    def _get_wind_direction(self, degrees):
        """Convert wind direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]

    async def _handle_image_generation(self, query):
        """FIXED: Handle standalone image generation with URL display"""
        image_prompt_match = re.search(
            r'(?:create\s+(?:an?\s+)?image\s+(?:of\s+)?|generate\s+(?:an?\s+)?image\s+(?:of\s+)?|'
            r'draw\s+(?:me\s+)?(?:an?\s+)?|show\s+(?:me\s+)?(?:an?\s+)?image\s+(?:of\s+)?|'
            r'make\s+(?:an?\s+)?image\s+(?:of\s+)?|picture\s+of\s+|image\s+of\s+|'
            r'generate\s+(?:a\s+)?picture\s+(?:of\s+)?|paint\s+(?:me\s+)?|sketch\s+|'
            r'artwork\s+of\s+|illustration\s+of\s+|visualize\s+)(.*)', 
            query, flags=re.I
        )
        
        if image_prompt_match:
            image_prompt = image_prompt_match.group(1).strip()
        else:
            image_prompt = query.replace("create image", "").replace("generate image", "").replace("draw", "").strip()

        if not image_prompt or len(image_prompt) < 3:
            image_prompt = "a beautiful landscape"
            
        result = await self._generate_image_from_text(image_prompt)
        
        if isinstance(result, dict) and result.get('status') == 'success':
            # FIXED: Clear display of image URL for standalone generation with clickable link
            response = f"✅ **Image Generated Successfully!**\n\n"
            response += f"🖼️ **Click to view image:** {result['image_url']}\n"
            response += f"🔗 **Direct link:** {result['image_url']}\n\n"
            response += f"📝 *Prompt: {result['prompt_used'][:100]}...*\n"
            response += f"💾 *Saved locally at: {result.get('local_path', 'N/A')}*"
            
            return {
                'type': 'image_generation',
                'text_response': response,
                'image_url': result['image_url'],
                'image_base64': result.get('image_base64'),
                'local_path': result.get('local_path'),
                'prompt_used': result['prompt_used']
            }
        else:
            # FIXED: Return proper error message instead of just success message
            error_msg = result.get('message', '❌ Image generation failed')
            return f"❌ **Image Generation Failed**\n\n{error_msg}\n\nPlease try again with a different prompt."

    async def _get_programming_help(self, query):
        """Get programming help"""
        prompt = f"""You are an expert programming assistant. Provide clear help for:

Question: {query}

Provide:
1. Clear explanation
2. Code examples if applicable
3. Best practices

Be practical and actionable."""
        
        return await self._call_gemini_text_model(prompt, self.conversation_history)

    async def _translate_text(self, query):
        """Translate text"""
        translation_prompt = f"""Please translate the following. Detect the source language:

{query}

Format:
- Source language: [detected]
- Translation: [translated]
- Notes: [relevant notes]"""
        
        return await self._call_gemini_text_model(translation_prompt, self.conversation_history)

    async def _get_knowledge(self, query):
        """Get knowledge"""
        knowledge_prompt = f"""Provide accurate information about: {query}

Focus on:
- Key facts and definitions
- Historical context if relevant
- Current developments
- Practical examples

Keep it informative but concise."""
        
        return await self._call_gemini_text_model(knowledge_prompt, self.conversation_history)

    def get_help(self):
        """Get help"""
        return """🤖 **Enhanced AI Assistant - Available Features:**

**💬 General Chat & Knowledge**
- Ask me anything

**📈 Financial Services**
- Stock prices: "What's the price of AAPL?"
- Portfolio: "Show me AAPL GOOGL TSLA portfolio"

**🎨 Image Generation**
- "create image sunset" (NO colon needed!)
- "generate image Donald Trump"
- "draw me a mountain"

**🔍 Monitoring & Alerts**
- Stock alerts: "monitor AAPL above 150"
- News monitoring: "monitor news about AI"
- List alerts: "list alerts"

**🌤️ Weather (Works for ALL cities worldwide!)**
- "weather in London"
- "weather in Lahore"
- "weather in Mumbai"
- "weather in Tokyo"
- Works for ANY city name!

**📰 News**
- "latest tech news"

**🔧 Other Features**
- Translation, Math, Programming help

Type any question to get started!"""

    def get_system_stats(self):
        """Get system stats"""
        active_monitors = sum(1 for task in self.monitoring_tasks.values() if task.is_active)
        total_conversations = len(self.conversation_history)
        
        recent_conversations = [conv for conv in self.conversation_history[-10:] 
                              if conv.get('confidence_score')]
        avg_confidence = sum(conv['confidence_score'] for conv in recent_conversations) / max(1, len(recent_conversations))
        
        return f"""📊 **System Status & Statistics**

**System Health**
- Status: Active and Running
- Database: Connected
- Background Services: Running

**Performance Metrics**
- Total Conversations: {total_conversations}
- Average Confidence: {avg_confidence:.1%}
- Active Monitors: {active_monitors}
- Cache Entries: {len(self.financial_cache)}

**API Status**
- Gemini AI: {'✅ Available' if self.google_generative_ai_api_key != 'YOUR_KEY' else '❌ Not Available'}
- Weather API: {'✅ Available' if self.openweathermap_api_key != 'YOUR_KEY' else '❌ Not Available'}
- News API: {'✅ Available' if self.newsapi_org_api_key != 'YOUR_KEY' else '❌ Not Available'}

*System uptime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""

    def _fallback_response(self, query):
        """Fallback response"""
        return f"""I encountered an issue processing: "{query}"

Try:
1. Rephrase your question
2. Check features with: help
3. Check system: status

Recent successful queries: stock prices, image generation, weather forecasts, general knowledge."""

async def main():
    bot = EnhancedFreeChatBot()
    print("✅ Enhanced AI Assistant Ready!")
    print("Type 'help' for features or 'quit' to exit\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'help':
            print(bot.get_help())
            continue
        
        if user_input.lower() in ['clear', 'reset']:
            bot.conversation_history = []
            print("Conversation history cleared.")
            continue
        
        if user_input.lower() in ['status', 'stats']:
            print(bot.get_system_stats())
            continue
        
        if not user_input:
            print("Please type something!")
            continue

        try:
            response = await bot.process_message(user_input)
            
            if isinstance(response, dict):
                if response.get('type') == 'image_generation':
                    print(f"\nAssistant: {response['text_response']}")
                    if response.get('image_base64'):
                        print("✅ Image data available for display")
                elif response.get('type') == 'multi_modal':
                    if response.get('combined_response'):
                        print(f"\nAssistant: {response['combined_response']}")
                    else:
                        print(f"\nAssistant: {response.get('text_response', '')}")
                        if response.get('image_data'):
                            print("✅ Image generated and available")
                else:
                    print(f"\nAssistant: {response}")
            else:
                print(f"\nAssistant: {response}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            logging.error(f"Main loop error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSession ended. Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")

        logging.error(f"Fatal error in main: {e}")
