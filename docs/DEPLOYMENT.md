# 🚀 Production Deployment Guide

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Platform Deployment](#cloud-platform-deployment)
5. [Database Setup](#database-setup)
6. [Environment Configuration](#environment-configuration)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Logging](#monitoring--logging)
9. [Troubleshooting](#troubleshooting)

---

## ✅ Pre-Deployment Checklist

### System Requirements

**Minimum Requirements**:
- OS: Ubuntu 20.04 LTS / Windows 10 / macOS 10.15+
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB available
- Network: Stable internet connection

**Recommended for Production**:
- OS: Ubuntu 22.04 LTS (Server)
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 100+ GB SSD
- Network: High-speed dedicated connection
- GPU: NVIDIA GPU (optional, for video processing)

### Required API Keys

Before deployment, obtain these API keys:

1. **Google Gemini API Key** (Required)
   - Get from: https://makersuite.google.com/app/apikey
   - Used for: Text processing and AI conversations

2. **OpenWeatherMap API Key** (Optional)
   - Get from: https://openweathermap.org/api
   - Used for: Weather information

3. **NewsAPI Key** (Optional)
   - Get from: https://newsapi.org/register
   - Used for: News aggregation

### Software Dependencies

```bash
# Python 3.8 or higher
python --version

# pip (Python package manager)
pip --version

# Git
git --version

# Docker (for containerized deployment)
docker --version
docker-compose --version
```

---

## 💻 Local Development Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/quantum-ai-nexus.git
cd quantum-ai-nexus
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
```

### Step 4: Set Up Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

Required environment variables:
```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# AI API Keys
GEMINI_API_KEY=your-gemini-api-key
OPENWEATHER_API_KEY=your-openweather-key
NEWSAPI_KEY=your-newsapi-key

# Database
DATABASE_URL=sqlite:///./database/app.db

# Session Configuration
SESSION_COOKIE_SECURE=False
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# Upload Configuration
MAX_CONTENT_LENGTH=20971520  # 20MB in bytes
UPLOAD_FOLDER=uploads

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Step 5: Initialize Database

```bash
# Create database directory
mkdir -p database

# Initialize database
python scripts/init_db.py
```

### Step 6: Run Application

```bash
# Start Flask application
python app.py

# Application will be available at:
# http://localhost:5000
```

### Step 7: Verify Installation

```bash
# Test health endpoint
curl http://localhost:5000/

# Expected response:
# {"status": "healthy", "version": "1.0.0"}
```

---

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build Docker image
docker build -t quantum-ai-nexus:latest .

# Run container
docker run -d \
  --name quantum-ai \
  -p 5000:5000 \
  -e GEMINI_API_KEY=your-key \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/database:/app/database \
  quantum-ai-nexus:latest
```

### Docker Compose (Recommended)

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: quantum-ai-nexus
    restart: always
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/quantum_ai
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - quantum-network

  db:
    image: postgres:14-alpine
    container_name: quantum-db
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=quantum_ai
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - quantum-network

  redis:
    image: redis:7-alpine
    container_name: quantum-redis
    restart: always
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - quantum-network

  nginx:
    image: nginx:alpine
    container_name: quantum-nginx
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - quantum-network

volumes:
  postgres_data:
  redis_data:

networks:
  quantum-network:
    driver: bridge
```

**Deploy with Docker Compose**:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

---

## ☁️ Cloud Platform Deployment

### AWS EC2 Deployment

#### Step 1: Launch EC2 Instance

```bash
# Launch Ubuntu Server 22.04 LTS
# Instance Type: t3.large (or larger)
# Configure Security Group:
#   - HTTP (80)
#   - HTTPS (443)
#   - SSH (22)
```

#### Step 2: Connect and Setup

```bash
# Connect via SSH
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/yourusername/quantum-ai-nexus.git
cd quantum-ai-nexus

# Set environment variables
nano .env

# Deploy
docker-compose up -d
```

#### Step 3: Configure Domain & SSL

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### Google Cloud Platform (GCP)

```bash
# Create Compute Engine instance
gcloud compute instances create quantum-ai-nexus \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=n1-standard-4 \
  --boot-disk-size=100GB

# SSH into instance
gcloud compute ssh quantum-ai-nexus

# Follow same setup as AWS above
```

### Microsoft Azure

```bash
# Create VM
az vm create \
  --resource-group quantum-ai-rg \
  --name quantum-ai-vm \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Open ports
az vm open-port --port 80 --resource-group quantum-ai-rg --name quantum-ai-vm
az vm open-port --port 443 --resource-group quantum-ai-rg --name quantum-ai-vm

# SSH and setup
ssh azureuser@<vm-ip>
# Follow same setup as AWS
```

### Heroku Deployment

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login

# Create app
heroku create quantum-ai-nexus

# Set environment variables
heroku config:set GEMINI_API_KEY=your-key
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# Scale dynos
heroku ps:scale web=2
```

---

## 🗄️ Database Setup

### SQLite (Development)

```python
# Automatic setup - no configuration needed
# Database file: database/app.db
```

### PostgreSQL (Production)

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Create database and user
sudo -u postgres psql
CREATE DATABASE quantum_ai;
CREATE USER quantum_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE quantum_ai TO quantum_user;
\q

# Update .env
DATABASE_URL=postgresql://quantum_user:your_password@localhost:5432/quantum_ai

# Run migrations
python scripts/migrate_db.py
```

### Database Backup

```bash
# PostgreSQL backup
pg_dump quantum_ai > backup_$(date +%Y%m%d).sql

# Restore
psql quantum_ai < backup_20240115.sql

# Automated daily backups
echo "0 2 * * * pg_dump quantum_ai > /backup/db_$(date +\%Y\%m\%d).sql" | crontab -
```

---

## ⚙️ Environment Configuration

### Production Environment Variables

```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=generate-strong-random-key-here

# AI API Keys
GEMINI_API_KEY=your-production-gemini-key
OPENWEATHER_API_KEY=your-production-weather-key
NEWSAPI_KEY=your-production-news-key

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:pass@localhost:5432/quantum_ai

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Security
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Strict
CSRF_ENABLED=True

# Performance
WORKERS=4
THREADS=2
WORKER_CLASS=sync
TIMEOUT=120

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/var/log/quantum-ai/app.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=10

# Rate Limiting
RATELIMIT_ENABLED=True
RATELIMIT_STORAGE_URL=redis://localhost:6379/1

# File Upload
MAX_CONTENT_LENGTH=20971520  # 20MB
UPLOAD_FOLDER=/var/uploads

# Monitoring
SENTRY_DSN=your-sentry-dsn  # Optional
```

### Generate Secret Key

```python
import secrets
print(secrets.token_hex(32))
```

---

## 🚀 Performance Optimization

### 1. Enable Production Mode

```python
# app.py
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### 2. Use Production WSGI Server

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn --workers 4 --threads 2 --bind 0.0.0.0:5000 app:app

# Or use systemd service (recommended)
```

**Create systemd service** (`/etc/systemd/system/quantum-ai.service`):
```ini
[Unit]
Description=Quantum AI Nexus
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/quantum-ai-nexus
Environment="PATH=/opt/quantum-ai-nexus/venv/bin"
ExecStart=/opt/quantum-ai-nexus/venv/bin/gunicorn \
    --workers 4 \
    --threads 2 \
    --timeout 120 \
    --bind 0.0.0.0:5000 \
    --access-logfile /var/log/quantum-ai/access.log \
    --error-logfile /var/log/quantum-ai/error.log \
    app:app

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable quantum-ai
sudo systemctl start quantum-ai
sudo systemctl status quantum-ai
```

### 3. Configure Nginx Reverse Proxy

**nginx.conf**:
```nginx
upstream quantum_ai {
    least_conn;
    server 127.0.0.1:5000;
    # Add more servers for load balancing
    # server 127.0.0.1:5001;
    # server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Max upload size
    client_max_body_size 20M;

    # Compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;

    # Static files
    location /static {
        alias /opt/quantum-ai-nexus/static;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # API endpoints
    location / {
        proxy_pass http://quantum_ai;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

### 4. Enable Redis Caching

```bash
# Install Redis
sudo apt install redis-server -y

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: maxmemory 2gb
# Set: maxmemory-policy allkeys-lru

# Restart Redis
sudo systemctl restart redis
```

---

## 📊 Monitoring & Logging

### Application Logging

```python
# Configure in app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10485760,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Quantum AI Nexus startup')
```

### System Monitoring

```bash
# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Monitor processes
htop

# Monitor network
nethogs

# Check logs
tail -f /var/log/quantum-ai/app.log
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### Application Performance Monitoring (APM)

```python
# Install Sentry for error tracking
pip install sentry-sdk[flask]

# Configure in app.py
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[FlaskIntegration()],
    traces_sample_rate=0.1
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
journalctl -u quantum-ai -n 50

# Verify Python dependencies
pip list

# Check port availability
sudo netstat -tulpn | grep 5000
```

#### 2. Database Connection Error

```bash
# Test database connection
psql -h localhost -U quantum_user -d quantum_ai

# Check database service
sudo systemctl status postgresql

# Verify DATABASE_URL in .env
```

#### 3. AI API Errors

```bash
# Verify API keys
echo $GEMINI_API_KEY

# Test API directly
curl https://generativelanguage.googleapis.com/v1/models?key=YOUR_KEY
```

#### 4. High Memory Usage

```bash
# Check memory
free -h

# Identify memory-heavy processes
ps aux --sort=-%mem | head

# Optimize worker count
# Reduce WORKERS in .env or gunicorn config
```

#### 5. Slow Response Times

```bash
# Enable query logging
# Check database query performance
# Verify Redis cache is working
redis-cli MONITOR

# Profile application
# Use cProfile or py-spy
```

### Health Check Endpoint

```python
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'database': check_database_connection(),
        'redis': check_redis_connection(),
        'ai_services': check_ai_services()
    })
```

### Rollback Procedure

```bash
# If deployment fails, rollback to previous version

# Docker
docker-compose down
git checkout previous-tag
docker-compose up -d

# Git
git reset --hard previous-commit
sudo systemctl restart quantum-ai
```

---

## 🎯 Post-Deployment Checklist

- [ ] Application accessible via domain
- [ ] SSL certificate installed and working
- [ ] Database backup configured
- [ ] Monitoring and logging in place
- [ ] Error tracking configured (Sentry)
- [ ] Performance optimized (Redis, Nginx)
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Health checks working
- [ ] Documentation updated

---

## 📞 Support

For deployment issues or questions:
- Email: abbaskhan0011ehe@gmail.com
- GitHub Issues: https://github.com/MuhammadAbbas01/quantum-ai-nexus/issues
- LinkedIn: [linkedin.com/in/muhammadabbas-ai](https://www.linkedin.com/in/muhammadabbas-ai/)

---

**Last Updated**: January 2024  
**Version**: 1.0.0
