-- db_init.sql
CREATE TABLE IF NOT EXISTS agent_performance_hourly (
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    agent_id VARCHAR(50),
    total_calls BIGINT,
    resolved_calls BIGINT,
    resolution_rate DOUBLE PRECISION,
    avg_call_duration DOUBLE PRECISION,
    avg_sentiment_score DOUBLE PRECISION,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (window_start, agent_id)
);

CREATE TABLE IF NOT EXISTS product_issue_trends_hourly (
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    product_category VARCHAR(100),
    issue_description VARCHAR(255),
    call_count BIGINT,
    avg_sentiment_category DOUBLE PRECISION,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (window_start, product_category, issue_description)
);