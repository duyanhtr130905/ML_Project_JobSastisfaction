-- Initialize Translation Database
-- Tables for analytics and metrics

-- Translation metrics table
CREATE TABLE IF NOT EXISTS translation_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_text TEXT NOT NULL,
    target_text TEXT NOT NULL,
    method VARCHAR(50),
    translation_time_ms FLOAT,
    cache_hit BOOLEAN DEFAULT FALSE,
    user_id VARCHAR(100),
    device_type VARCHAR(50),
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster queries
CREATE INDEX idx_translation_metrics_timestamp ON translation_metrics(timestamp);
CREATE INDEX idx_translation_metrics_user ON translation_metrics(user_id);
CREATE INDEX idx_translation_metrics_method ON translation_metrics(method);

-- Terminology usage table
CREATE TABLE IF NOT EXISTS terminology_usage (
    id SERIAL PRIMARY KEY,
    term_en VARCHAR(255) NOT NULL,
    term_vi VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for terminology
CREATE INDEX idx_terminology_term_en ON terminology_usage(term_en);
CREATE INDEX idx_terminology_category ON terminology_usage(category);

-- User activity table
CREATE TABLE IF NOT EXISTS user_activity (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    details JSONB,
    ip_address VARCHAR(50),
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for user activity
CREATE INDEX idx_user_activity_user ON user_activity(user_id);
CREATE INDEX idx_user_activity_timestamp ON user_activity(timestamp);
CREATE INDEX idx_user_activity_action ON user_activity(action);

-- Translation quality feedback table
CREATE TABLE IF NOT EXISTS translation_feedback (
    id SERIAL PRIMARY KEY,
    translation_id INTEGER,
    user_id VARCHAR(100),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for system metrics
CREATE INDEX idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create views for common queries

-- Daily translation summary
CREATE OR REPLACE VIEW daily_translation_summary AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_translations,
    AVG(translation_time_ms) as avg_time_ms,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
    SUM(CASE WHEN method = 'memory_exact' THEN 1 ELSE 0 END) as memory_translations,
    SUM(CASE WHEN method = 'base_translation' THEN 1 ELSE 0 END) as new_translations,
    COUNT(DISTINCT user_id) as unique_users
FROM translation_metrics
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Top users by translation volume
CREATE OR REPLACE VIEW top_users_by_volume AS
SELECT 
    user_id,
    COUNT(*) as translation_count,
    AVG(translation_time_ms) as avg_time_ms,
    MAX(timestamp) as last_active
FROM translation_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id
ORDER BY translation_count DESC
LIMIT 100;

-- Most used terminology
CREATE OR REPLACE VIEW top_terminology AS
SELECT 
    term_en,
    term_vi,
    category,
    usage_count,
    last_used
FROM terminology_usage
ORDER BY usage_count DESC, last_used DESC
LIMIT 100;

-- Insert some sample data (optional)
-- Uncomment to insert sample data for testing

-- INSERT INTO translation_metrics (source_text, target_text, method, translation_time_ms, cache_hit, user_id, device_type)
-- VALUES 
-- ('Hello world', 'Xin chào thế giới', 'base_translation', 150.5, FALSE, 'user1', 'web'),
-- ('Machine learning', 'Học máy', 'memory_exact', 50.2, TRUE, 'user1', 'web'),
-- ('Artificial intelligence', 'Trí tuệ nhân tạo', 'base_translation', 180.3, FALSE, 'user2', 'mobile');

-- INSERT INTO terminology_usage (term_en, term_vi, category, usage_count)
-- VALUES 
-- ('algorithm', 'thuật toán', 'computer_science', 10),
-- ('machine learning', 'học máy', 'ai', 25),
-- ('neural network', 'mạng nơ-ron', 'ai', 15);

COMMENT ON TABLE translation_metrics IS 'Stores metrics for each translation request';
COMMENT ON TABLE terminology_usage IS 'Tracks usage of scientific terminology';
COMMENT ON TABLE user_activity IS 'Logs user actions and interactions';
COMMENT ON TABLE translation_feedback IS 'Stores user feedback on translation quality';
COMMENT ON TABLE system_metrics IS 'System performance and health metrics';
