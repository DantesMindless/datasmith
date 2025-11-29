-- Machine Learning-ready datasets for PostgreSQL test environment
-- Day 1-2: Create test databases with ML-friendly data

-- Database 1: Customer Churn Prediction (Classification)
CREATE DATABASE customer_churn;

\c customer_churn;

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    tenure_months INT,
    monthly_charges DECIMAL(10, 2),
    total_charges DECIMAL(10, 2),
    contract_type VARCHAR(50), -- Month-to-month, One year, Two year
    payment_method VARCHAR(50),
    internet_service VARCHAR(50), -- DSL, Fiber optic, No
    online_security VARCHAR(10), -- Yes, No
    tech_support VARCHAR(10), -- Yes, No
    streaming_tv VARCHAR(10), -- Yes, No
    streaming_movies VARCHAR(10), -- Yes, No
    paperless_billing VARCHAR(10), -- Yes, No
    num_support_tickets INT DEFAULT 0,
    num_late_payments INT DEFAULT 0,
    churn BOOLEAN -- TARGET: Did customer leave? (for classification)
);

-- Insert realistic customer churn data (good for ML training)
INSERT INTO customers (age, gender, tenure_months, monthly_charges, total_charges, contract_type, payment_method, internet_service, online_security, tech_support, streaming_tv, streaming_movies, paperless_billing, num_support_tickets, num_late_payments, churn) VALUES
(45, 'Male', 24, 89.99, 2159.76, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 1, 0, FALSE),
(32, 'Female', 3, 55.50, 166.50, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 5, 2, TRUE),
(28, 'Male', 12, 70.25, 843.00, 'One year', 'Bank transfer', 'Fiber optic', 'Yes', 'No', 'Yes', 'No', 'No', 2, 0, FALSE),
(52, 'Female', 6, 45.00, 270.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 8, 3, TRUE),
(38, 'Male', 36, 95.75, 3447.00, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 0, 0, FALSE),
(61, 'Female', 48, 88.50, 4248.00, 'Two year', 'Bank transfer', 'Fiber optic', 'Yes', 'Yes', 'No', 'Yes', 'No', 1, 0, FALSE),
(25, 'Male', 2, 50.00, 100.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 7, 4, TRUE),
(44, 'Female', 18, 75.00, 1350.00, 'One year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 2, 1, FALSE),
(29, 'Male', 1, 40.00, 40.00, 'Month-to-month', 'Mailed check', 'No', 'No', 'No', 'No', 'No', 'No', 3, 1, TRUE),
(55, 'Female', 60, 99.99, 5999.40, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 0, 0, FALSE),
(33, 'Male', 8, 65.00, 520.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'Yes', 'No', 'No', 'Yes', 4, 2, TRUE),
(47, 'Female', 24, 80.00, 1920.00, 'One year', 'Bank transfer', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 1, 0, FALSE),
(26, 'Male', 4, 55.00, 220.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 6, 3, TRUE),
(50, 'Female', 42, 92.00, 3864.00, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 0, 0, FALSE),
(35, 'Male', 15, 68.50, 1027.50, 'One year', 'Bank transfer', 'DSL', 'Yes', 'No', 'Yes', 'No', 'No', 3, 1, FALSE),
(27, 'Female', 2, 48.00, 96.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 9, 5, TRUE),
(41, 'Male', 30, 85.00, 2550.00, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 1, 0, FALSE),
(58, 'Female', 54, 97.50, 5265.00, 'Two year', 'Bank transfer', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 0, 0, FALSE),
(24, 'Male', 1, 42.00, 42.00, 'Month-to-month', 'Mailed check', 'No', 'No', 'No', 'No', 'No', 'No', 5, 2, TRUE),
(49, 'Female', 20, 77.00, 1540.00, 'One year', 'Credit card', 'Fiber optic', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 2, 0, FALSE),
(31, 'Male', 5, 58.00, 290.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 7, 4, TRUE),
(53, 'Female', 36, 90.00, 3240.00, 'Two year', 'Bank transfer', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'No', 'No', 0, 0, FALSE),
(28, 'Male', 3, 52.00, 156.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'No', 'No', 'No', 'Yes', 8, 3, TRUE),
(46, 'Female', 28, 82.50, 2310.00, 'Two year', 'Credit card', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 1, 0, FALSE),
(36, 'Male', 10, 62.00, 620.00, 'Month-to-month', 'Electronic check', 'DSL', 'No', 'Yes', 'No', 'No', 'Yes', 5, 2, TRUE);

-- Database 2: House Price Prediction (Regression)
CREATE DATABASE housing_prices;

\c housing_prices;

CREATE TABLE houses (
    house_id SERIAL PRIMARY KEY,
    square_feet INT,
    num_bedrooms INT,
    num_bathrooms DECIMAL(3, 1),
    lot_size_sqft INT,
    year_built INT,
    garage_spaces INT,
    neighborhood VARCHAR(100),
    has_pool BOOLEAN,
    has_fireplace BOOLEAN,
    condition_rating INT, -- 1-5 scale
    school_rating INT, -- 1-10 scale
    distance_to_city_center_miles DECIMAL(5, 2),
    crime_rate DECIMAL(5, 2), -- per 1000 residents
    price DECIMAL(12, 2) -- TARGET: House price (for regression)
);

-- Insert realistic housing data
INSERT INTO houses (square_feet, num_bedrooms, num_bathrooms, lot_size_sqft, year_built, garage_spaces, neighborhood, has_pool, has_fireplace, condition_rating, school_rating, distance_to_city_center_miles, crime_rate, price) VALUES
(2400, 4, 2.5, 8000, 2015, 2, 'Riverside', TRUE, TRUE, 5, 9, 3.5, 2.1, 485000.00),
(1200, 2, 1.0, 4000, 1985, 1, 'Downtown', FALSE, FALSE, 3, 6, 0.5, 8.5, 225000.00),
(3200, 5, 3.5, 12000, 2018, 3, 'Hillside', TRUE, TRUE, 5, 10, 8.2, 1.2, 725000.00),
(1800, 3, 2.0, 6000, 2000, 2, 'Suburbia', FALSE, TRUE, 4, 7, 5.0, 3.5, 340000.00),
(1500, 3, 1.5, 5000, 1990, 1, 'Oldtown', FALSE, FALSE, 3, 5, 2.0, 6.8, 198000.00),
(2800, 4, 3.0, 10000, 2020, 2, 'Lakeside', TRUE, TRUE, 5, 9, 6.5, 1.8, 595000.00),
(1000, 2, 1.0, 3500, 1975, 1, 'Industrial', FALSE, FALSE, 2, 4, 4.5, 12.3, 145000.00),
(2200, 4, 2.5, 7500, 2010, 2, 'Parkview', FALSE, TRUE, 4, 8, 4.0, 2.8, 425000.00),
(3500, 5, 4.0, 15000, 2021, 3, 'Estates', TRUE, TRUE, 5, 10, 10.0, 0.8, 850000.00),
(1600, 3, 2.0, 5500, 1995, 2, 'Meadows', FALSE, TRUE, 4, 7, 6.0, 4.2, 285000.00),
(2000, 3, 2.5, 7000, 2012, 2, 'Riverside', FALSE, TRUE, 4, 9, 3.8, 2.3, 395000.00),
(1300, 2, 1.5, 4500, 1988, 1, 'Downtown', FALSE, FALSE, 3, 6, 1.0, 7.5, 245000.00),
(2600, 4, 3.0, 9000, 2016, 2, 'Hillside', TRUE, TRUE, 5, 10, 7.5, 1.5, 625000.00),
(1900, 3, 2.0, 6500, 2005, 2, 'Suburbia', FALSE, TRUE, 4, 7, 5.5, 3.2, 365000.00),
(1100, 2, 1.0, 3800, 1980, 1, 'Oldtown', FALSE, FALSE, 2, 5, 2.5, 8.0, 175000.00),
(3000, 5, 3.5, 11000, 2019, 3, 'Lakeside', TRUE, TRUE, 5, 9, 7.0, 1.6, 685000.00),
(900, 1, 1.0, 3000, 1970, 0, 'Industrial', FALSE, FALSE, 2, 3, 5.0, 15.2, 125000.00),
(2500, 4, 2.5, 8500, 2014, 2, 'Parkview', FALSE, TRUE, 4, 8, 4.5, 2.5, 465000.00),
(4000, 6, 4.5, 18000, 2022, 3, 'Estates', TRUE, TRUE, 5, 10, 11.0, 0.5, 975000.00),
(1750, 3, 2.0, 6000, 1998, 2, 'Meadows', FALSE, TRUE, 4, 7, 6.5, 3.8, 315000.00),
(2100, 4, 2.5, 7200, 2008, 2, 'Riverside', FALSE, TRUE, 4, 9, 4.0, 2.6, 415000.00),
(1400, 2, 1.5, 4800, 1992, 1, 'Downtown', FALSE, FALSE, 3, 6, 0.8, 7.2, 265000.00),
(2900, 5, 3.5, 10500, 2017, 3, 'Hillside', TRUE, TRUE, 5, 10, 8.0, 1.3, 695000.00),
(1850, 3, 2.0, 6200, 2003, 2, 'Suburbia', FALSE, TRUE, 4, 7, 5.2, 3.6, 355000.00),
(1250, 2, 1.5, 4200, 1986, 1, 'Oldtown', FALSE, FALSE, 3, 5, 2.2, 7.8, 205000.00);

-- Database 3: Credit Card Fraud Detection (Classification)
CREATE DATABASE fraud_detection;

\c fraud_detection;

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    transaction_amount DECIMAL(10, 2),
    transaction_hour INT, -- 0-23
    day_of_week INT, -- 0-6
    merchant_category VARCHAR(50),
    card_present BOOLEAN,
    online_transaction BOOLEAN,
    international BOOLEAN,
    distance_from_home_km DECIMAL(8, 2),
    distance_from_last_transaction_km DECIMAL(8, 2),
    time_since_last_transaction_hours DECIMAL(8, 2),
    num_transactions_24h INT,
    avg_amount_30d DECIMAL(10, 2),
    is_fraud BOOLEAN -- TARGET: Fraudulent transaction? (for classification)
);

-- Insert realistic fraud detection data
INSERT INTO transactions (transaction_amount, transaction_hour, day_of_week, merchant_category, card_present, online_transaction, international, distance_from_home_km, distance_from_last_transaction_km, time_since_last_transaction_hours, num_transactions_24h, avg_amount_30d, is_fraud) VALUES
(45.50, 14, 3, 'Grocery', TRUE, FALSE, FALSE, 2.5, 1.2, 3.5, 2, 52.30, FALSE),
(1250.00, 3, 2, 'Electronics', FALSE, TRUE, TRUE, 8500.00, 8500.00, 0.5, 15, 45.00, TRUE),
(85.00, 18, 5, 'Restaurant', TRUE, FALSE, FALSE, 5.0, 3.2, 2.0, 3, 68.50, FALSE),
(2500.00, 2, 1, 'Jewelry', FALSE, TRUE, TRUE, 5200.00, 5200.00, 1.0, 20, 50.00, TRUE),
(32.75, 12, 4, 'Gas Station', TRUE, FALSE, FALSE, 8.0, 5.5, 24.0, 1, 55.20, FALSE),
(150.00, 20, 6, 'Clothing', FALSE, TRUE, FALSE, 0.5, 0.5, 4.0, 2, 75.00, FALSE),
(3200.00, 4, 0, 'Electronics', FALSE, TRUE, TRUE, 9800.00, 9800.00, 0.2, 25, 48.00, TRUE),
(67.50, 11, 2, 'Pharmacy', TRUE, FALSE, FALSE, 3.0, 2.0, 12.0, 1, 62.00, FALSE),
(1800.00, 1, 3, 'Luxury Goods', FALSE, TRUE, TRUE, 6500.00, 6500.00, 0.8, 18, 52.00, TRUE),
(95.00, 19, 5, 'Entertainment', TRUE, FALSE, FALSE, 12.0, 10.0, 6.0, 2, 71.00, FALSE),
(28.50, 8, 1, 'Coffee Shop', TRUE, FALSE, FALSE, 1.5, 1.0, 16.0, 1, 58.00, FALSE),
(2100.00, 3, 4, 'Online Marketplace', FALSE, TRUE, TRUE, 7200.00, 7200.00, 0.3, 22, 46.00, TRUE),
(120.00, 13, 3, 'Supermarket', TRUE, FALSE, FALSE, 4.0, 3.5, 8.0, 2, 65.00, FALSE),
(4500.00, 2, 6, 'Electronics', FALSE, TRUE, TRUE, 11000.00, 11000.00, 0.4, 30, 50.00, TRUE),
(55.00, 17, 2, 'Bookstore', TRUE, FALSE, FALSE, 6.0, 4.0, 5.0, 2, 60.00, FALSE),
(75.00, 15, 4, 'Department Store', FALSE, TRUE, FALSE, 1.0, 0.8, 3.0, 1, 70.00, FALSE),
(3800.00, 4, 1, 'Jewelry', FALSE, TRUE, TRUE, 8900.00, 8900.00, 0.6, 28, 47.00, TRUE),
(42.00, 10, 5, 'Fast Food', TRUE, FALSE, FALSE, 7.0, 5.0, 10.0, 2, 63.00, FALSE),
(1600.00, 1, 2, 'Online Gaming', FALSE, TRUE, TRUE, 4800.00, 4800.00, 1.2, 16, 51.00, TRUE),
(88.50, 16, 6, 'Home Improvement', TRUE, FALSE, FALSE, 9.0, 7.0, 7.0, 1, 67.00, FALSE),
(52.00, 9, 3, 'Convenience Store', TRUE, FALSE, FALSE, 2.0, 1.5, 14.0, 1, 56.00, FALSE),
(2900.00, 3, 5, 'Luxury Goods', FALSE, TRUE, TRUE, 7800.00, 7800.00, 0.5, 24, 49.00, TRUE),
(110.00, 21, 1, 'Sports Equipment', FALSE, TRUE, FALSE, 3.5, 2.5, 4.5, 2, 72.00, FALSE),
(5200.00, 2, 4, 'Electronics', FALSE, TRUE, TRUE, 12500.00, 12500.00, 0.3, 35, 45.00, TRUE),
(63.00, 12, 2, 'Pet Store', TRUE, FALSE, FALSE, 5.5, 4.0, 9.0, 1, 61.00, FALSE);

-- Add some views for common ML queries
\c customer_churn;
CREATE VIEW high_risk_customers AS
SELECT * FROM customers
WHERE contract_type = 'Month-to-month'
  AND num_support_tickets > 3
  AND tenure_months < 6;

\c housing_prices;
CREATE VIEW premium_houses AS
SELECT * FROM houses
WHERE price > 500000
  AND condition_rating >= 4;

\c fraud_detection;
CREATE VIEW suspicious_transactions AS
SELECT * FROM transactions
WHERE international = TRUE
  AND transaction_amount > 1000
  AND time_since_last_transaction_hours < 2;
