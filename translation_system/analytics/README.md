# Apache Superset Analytics Configuration

This directory contains configurations for Apache Superset integration to analyze translation performance in real-time.

## Overview

Apache Superset provides powerful analytics and visualization capabilities for:
- Translation performance metrics
- Usage statistics
- Quality trends
- User activity tracking
- System performance monitoring

## Setup

### Prerequisites

- Apache Superset installed (via Docker or native installation)
- PostgreSQL/MySQL database for storing translation metrics
- Access to translation system API

### Installation

1. **Start Superset with Docker**:
```bash
docker-compose -f docker-compose-superset.yml up -d
```

2. **Initialize Superset**:
```bash
docker exec -it superset superset db upgrade
docker exec -it superset superset fab create-admin \
    --username admin \
    --firstname Admin \
    --lastname User \
    --email admin@example.com \
    --password admin

docker exec -it superset superset init
```

3. **Access Superset**:
Open http://localhost:8088 in your browser and login with admin credentials.

## Database Connection

### Add Translation Database

1. Go to **Data > Databases**
2. Click **+ Database**
3. Configure connection:
   - **Database**: PostgreSQL / MySQL
   - **SQLAlchemy URI**: `postgresql://user:password@host:5432/translation_db`
4. Test connection and save

## Dashboards

### Translation Performance Dashboard

Tracks key metrics:
- Total translations per day/week/month
- Average translation time
- Cache hit rate (translations from memory)
- Terminology standardization rate
- Error rate

### User Activity Dashboard

Monitors:
- Active users
- Top translated documents
- Most used terminology
- Geographic distribution
- Device types (web/mobile)

### Quality Metrics Dashboard

Analyzes:
- Translation quality scores
- User feedback ratings
- Terminology consistency
- Translation corrections
- Similar translation matches

## Metrics Collection

### Database Schema

Create tables for metrics:

```sql
-- Translation metrics table
CREATE TABLE translation_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    source_text TEXT,
    target_text TEXT,
    method VARCHAR(50),
    translation_time FLOAT,
    cache_hit BOOLEAN,
    user_id VARCHAR(100),
    device_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Terminology usage table
CREATE TABLE terminology_usage (
    id SERIAL PRIMARY KEY,
    term_en VARCHAR(255),
    term_vi VARCHAR(255),
    category VARCHAR(100),
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User activity table
CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(50),
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### API Logging

Update the FastAPI backend to log metrics:

```python
# Add to api/main.py
from datetime import datetime
import psycopg2

def log_translation_metric(source, target, method, time_ms, cache_hit, user_id):
    conn = psycopg2.connect("postgresql://user:pass@host/db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO translation_metrics 
        (timestamp, source_text, target_text, method, translation_time, cache_hit, user_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (datetime.now(), source, target, method, time_ms, cache_hit, user_id))
    conn.commit()
    cur.close()
    conn.close()
```

## Charts and Visualizations

### Recommended Charts

1. **Translations Over Time** (Line Chart)
   - X-axis: Date
   - Y-axis: Number of translations
   - Metrics: Total, Cache hits, New translations

2. **Translation Method Distribution** (Pie Chart)
   - Segments: Memory exact, Base translation, Terminology-enhanced

3. **Average Response Time** (Bar Chart)
   - X-axis: Hour of day / Day of week
   - Y-axis: Average milliseconds

4. **Top Terminology Used** (Table)
   - Columns: Term, Category, Usage count, Last used

5. **User Activity Heatmap** (Heatmap)
   - X-axis: Hour
   - Y-axis: Day of week
   - Color: Number of translations

## Alerts and Monitoring

Configure alerts for:
- High error rates (> 5%)
- Slow response times (> 2 seconds average)
- Low cache hit rate (< 50%)
- System downtime

## Custom SQL Queries

### Daily Translation Summary
```sql
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_translations,
    AVG(translation_time) as avg_time_ms,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits
FROM translation_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### Top Users by Translation Volume
```sql
SELECT 
    user_id,
    COUNT(*) as translation_count,
    AVG(translation_time) as avg_time
FROM translation_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY user_id
ORDER BY translation_count DESC
LIMIT 10;
```

## Exporting Dashboards

1. Export dashboard as JSON:
   - Dashboard > ⋮ > Export
   - Save JSON file

2. Import on another instance:
   - Dashboard > ⋮ > Import
   - Upload JSON file

## Maintenance

- Regularly archive old metrics data
- Monitor database size and performance
- Update dashboards based on user feedback
- Review and optimize slow queries

## Support

For issues or customization requests, contact the development team.
