-- Machine Learning-ready datasets for MySQL test environment
-- Day 1-2: User Activity & Behavior Analytics

-- Database 1: User Activity & Engagement (Classification)
CREATE DATABASE IF NOT EXISTS user_engagement;
USE user_engagement;

CREATE TABLE user_activity (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    account_age_days INT,
    total_sessions INT,
    avg_session_duration_minutes DECIMAL(8, 2),
    pages_per_session DECIMAL(6, 2),
    total_logins INT,
    days_since_last_login INT,
    feature_usage_count INT,
    mobile_access_percentage DECIMAL(5, 2),
    desktop_access_percentage DECIMAL(5, 2),
    weekend_usage_percentage DECIMAL(5, 2),
    content_created INT,
    content_shared INT,
    comments_posted INT,
    profile_completeness DECIMAL(5, 2), -- 0-100
    has_profile_picture BOOLEAN,
    email_verified BOOLEAN,
    phone_verified BOOLEAN,
    active_user BOOLEAN -- TARGET: Is user active? (for classification)
);

-- Insert realistic user engagement data
INSERT INTO user_activity (age, account_age_days, total_sessions, avg_session_duration_minutes, pages_per_session, total_logins, days_since_last_login, feature_usage_count, mobile_access_percentage, desktop_access_percentage, weekend_usage_percentage, content_created, content_shared, comments_posted, profile_completeness, has_profile_picture, email_verified, phone_verified, active_user) VALUES
(28, 450, 320, 25.5, 8.2, 280, 1, 45, 65.0, 35.0, 22.0, 28, 15, 85, 95.0, TRUE, TRUE, TRUE, TRUE),
(35, 120, 15, 5.2, 2.1, 12, 45, 8, 80.0, 20.0, 15.0, 2, 0, 5, 40.0, FALSE, TRUE, FALSE, FALSE),
(42, 680, 520, 32.8, 12.5, 450, 0, 78, 45.0, 55.0, 28.0, 52, 38, 142, 100.0, TRUE, TRUE, TRUE, TRUE),
(22, 90, 8, 3.5, 1.8, 6, 60, 4, 95.0, 5.0, 10.0, 0, 0, 2, 25.0, FALSE, FALSE, FALSE, FALSE),
(31, 380, 280, 22.0, 7.5, 240, 2, 38, 70.0, 30.0, 25.0, 18, 12, 65, 88.0, TRUE, TRUE, TRUE, TRUE),
(45, 550, 420, 28.5, 9.8, 380, 1, 62, 40.0, 60.0, 20.0, 35, 22, 98, 92.0, TRUE, TRUE, TRUE, TRUE),
(25, 60, 5, 2.8, 1.5, 4, 75, 3, 100.0, 0.0, 8.0, 0, 0, 1, 20.0, FALSE, TRUE, FALSE, FALSE),
(38, 720, 580, 35.2, 14.0, 520, 0, 85, 50.0, 50.0, 30.0, 65, 45, 175, 100.0, TRUE, TRUE, TRUE, TRUE),
(29, 180, 45, 8.5, 3.2, 38, 15, 15, 75.0, 25.0, 18.0, 5, 2, 12, 60.0, TRUE, TRUE, FALSE, FALSE),
(52, 820, 650, 38.0, 15.5, 590, 0, 92, 35.0, 65.0, 18.0, 78, 55, 210, 100.0, TRUE, TRUE, TRUE, TRUE),
(26, 95, 10, 4.0, 1.9, 8, 50, 6, 90.0, 10.0, 12.0, 1, 0, 3, 35.0, FALSE, TRUE, FALSE, FALSE),
(33, 420, 310, 24.0, 8.5, 270, 1, 48, 60.0, 40.0, 24.0, 25, 18, 72, 90.0, TRUE, TRUE, TRUE, TRUE),
(48, 600, 480, 30.0, 11.2, 430, 1, 68, 42.0, 58.0, 22.0, 42, 28, 125, 95.0, TRUE, TRUE, TRUE, TRUE),
(23, 75, 6, 3.0, 1.6, 5, 65, 4, 100.0, 0.0, 5.0, 0, 0, 1, 22.0, FALSE, FALSE, FALSE, FALSE),
(36, 490, 370, 26.5, 9.0, 320, 2, 52, 55.0, 45.0, 26.0, 32, 20, 88, 93.0, TRUE, TRUE, TRUE, TRUE),
(41, 520, 400, 27.8, 10.5, 360, 1, 58, 48.0, 52.0, 21.0, 38, 25, 105, 94.0, TRUE, TRUE, TRUE, TRUE),
(27, 110, 12, 4.5, 2.0, 10, 55, 7, 85.0, 15.0, 14.0, 2, 1, 4, 38.0, FALSE, TRUE, FALSE, FALSE),
(50, 750, 600, 36.0, 13.8, 540, 0, 88, 38.0, 62.0, 19.0, 72, 48, 195, 100.0, TRUE, TRUE, TRUE, TRUE),
(30, 220, 80, 12.0, 4.5, 65, 10, 22, 68.0, 32.0, 20.0, 8, 4, 25, 65.0, TRUE, TRUE, FALSE, FALSE),
(44, 580, 450, 29.5, 11.8, 410, 1, 65, 44.0, 56.0, 23.0, 45, 32, 132, 96.0, TRUE, TRUE, TRUE, TRUE),
(24, 85, 8, 3.2, 1.7, 6, 70, 5, 95.0, 5.0, 9.0, 1, 0, 2, 28.0, FALSE, TRUE, FALSE, FALSE),
(37, 460, 340, 25.8, 8.8, 300, 2, 50, 58.0, 42.0, 25.0, 28, 19, 78, 91.0, TRUE, TRUE, TRUE, TRUE),
(46, 640, 510, 32.5, 12.8, 470, 0, 75, 40.0, 60.0, 20.0, 58, 40, 158, 98.0, TRUE, TRUE, TRUE, TRUE),
(28, 150, 25, 6.5, 2.8, 20, 30, 12, 78.0, 22.0, 16.0, 4, 2, 8, 55.0, TRUE, TRUE, FALSE, FALSE),
(39, 510, 390, 28.0, 10.2, 350, 1, 60, 52.0, 48.0, 24.0, 35, 24, 95, 94.0, TRUE, TRUE, TRUE, TRUE);

-- Database 2: Product Recommendations (Regression - predict rating)
CREATE DATABASE IF NOT EXISTS product_ratings;
USE product_ratings;

CREATE TABLE product_reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    user_age INT,
    user_account_age_days INT,
    user_previous_reviews INT,
    product_category VARCHAR(50),
    product_price DECIMAL(10, 2),
    product_age_days INT,
    product_total_reviews INT,
    product_avg_rating DECIMAL(3, 2),
    purchase_verified BOOLEAN,
    review_length_words INT,
    review_has_images BOOLEAN,
    review_helpful_votes INT,
    days_since_purchase INT,
    user_avg_rating DECIMAL(3, 2),
    rating DECIMAL(2, 1) -- TARGET: User rating (for regression)
);

-- Insert realistic product review data
INSERT INTO product_reviews (user_age, user_account_age_days, user_previous_reviews, product_category, product_price, product_age_days, product_total_reviews, product_avg_rating, purchase_verified, review_length_words, review_has_images, review_helpful_votes, days_since_purchase, user_avg_rating, rating) VALUES
(32, 450, 25, 'Electronics', 299.99, 180, 1250, 4.3, TRUE, 145, TRUE, 28, 15, 4.1, 5.0),
(45, 280, 8, 'Home & Garden', 89.99, 60, 420, 3.8, TRUE, 52, FALSE, 5, 8, 3.9, 4.0),
(28, 720, 42, 'Books', 24.99, 300, 850, 4.5, TRUE, 220, FALSE, 45, 45, 4.3, 5.0),
(55, 120, 3, 'Clothing', 45.50, 90, 180, 3.2, FALSE, 28, FALSE, 1, 5, 3.5, 2.0),
(38, 580, 35, 'Electronics', 599.99, 120, 2200, 4.6, TRUE, 180, TRUE, 52, 22, 4.4, 5.0),
(42, 340, 15, 'Sports', 125.00, 45, 650, 4.1, TRUE, 95, TRUE, 18, 12, 4.0, 4.0),
(25, 90, 2, 'Beauty', 35.99, 30, 95, 3.5, TRUE, 35, FALSE, 2, 3, 3.6, 3.0),
(50, 820, 58, 'Books', 18.99, 450, 1800, 4.7, TRUE, 280, FALSE, 68, 90, 4.5, 5.0),
(30, 180, 6, 'Kitchen', 149.99, 75, 520, 4.0, TRUE, 72, TRUE, 12, 18, 3.8, 4.0),
(48, 650, 48, 'Electronics', 899.99, 200, 3500, 4.4, TRUE, 195, TRUE, 75, 35, 4.2, 5.0),
(26, 95, 1, 'Clothing', 29.99, 20, 45, 3.0, FALSE, 18, FALSE, 0, 2, 3.2, 2.0),
(52, 880, 65, 'Home & Garden', 225.00, 240, 980, 4.2, TRUE, 165, TRUE, 42, 55, 4.3, 5.0),
(35, 420, 22, 'Sports', 78.50, 90, 420, 3.9, TRUE, 88, FALSE, 15, 25, 4.0, 4.0),
(29, 150, 5, 'Beauty', 52.00, 40, 220, 3.7, TRUE, 45, TRUE, 8, 10, 3.7, 3.0),
(44, 560, 38, 'Books', 32.50, 380, 1200, 4.6, TRUE, 245, FALSE, 58, 68, 4.4, 5.0),
(38, 380, 18, 'Kitchen', 189.99, 110, 780, 4.1, TRUE, 102, TRUE, 22, 28, 4.0, 4.0),
(27, 110, 4, 'Electronics', 129.99, 50, 280, 3.6, FALSE, 32, FALSE, 3, 6, 3.5, 3.0),
(51, 750, 55, 'Home & Garden', 175.00, 220, 850, 4.3, TRUE, 175, TRUE, 48, 50, 4.2, 5.0),
(33, 280, 12, 'Clothing', 68.00, 65, 320, 3.8, TRUE, 62, FALSE, 10, 15, 3.9, 4.0),
(46, 620, 42, 'Sports', 245.00, 150, 1100, 4.2, TRUE, 135, TRUE, 35, 32, 4.1, 5.0),
(24, 85, 2, 'Beauty', 28.50, 25, 85, 3.3, TRUE, 25, FALSE, 1, 4, 3.4, 2.0),
(49, 720, 52, 'Books', 42.00, 420, 1650, 4.5, TRUE, 268, FALSE, 62, 82, 4.4, 5.0),
(36, 450, 28, 'Kitchen', 95.00, 85, 580, 4.0, TRUE, 78, TRUE, 18, 22, 4.0, 4.0),
(28, 125, 3, 'Electronics', 85.00, 35, 150, 3.4, FALSE, 28, FALSE, 2, 5, 3.5, 2.0),
(40, 580, 35, 'Home & Garden', 135.00, 180, 720, 4.1, TRUE, 145, TRUE, 32, 42, 4.1, 4.0);

-- Database 3: Employee Attrition Prediction (Classification)
CREATE DATABASE IF NOT EXISTS employee_attrition;
USE employee_attrition;

CREATE TABLE employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    years_at_company INT,
    years_since_last_promotion INT,
    monthly_salary DECIMAL(10, 2),
    job_level INT, -- 1-5
    department VARCHAR(50),
    num_projects INT,
    avg_hours_per_week DECIMAL(5, 2),
    num_trainings_last_year INT,
    satisfaction_score DECIMAL(3, 2), -- 0-5
    performance_rating DECIMAL(3, 2), -- 0-5
    work_life_balance_score DECIMAL(3, 2), -- 0-5
    commute_distance_km DECIMAL(5, 2),
    remote_work_days_per_week INT,
    has_received_promotion BOOLEAN,
    num_managers INT,
    team_size INT,
    attrition BOOLEAN -- TARGET: Did employee leave? (for classification)
);

-- Insert realistic employee attrition data
INSERT INTO employees (age, years_at_company, years_since_last_promotion, monthly_salary, job_level, department, num_projects, avg_hours_per_week, num_trainings_last_year, satisfaction_score, performance_rating, work_life_balance_score, commute_distance_km, remote_work_days_per_week, has_received_promotion, num_managers, team_size, attrition) VALUES
(35, 8, 2, 8500.00, 3, 'Engineering', 4, 42.5, 3, 4.2, 4.5, 4.0, 12.5, 2, TRUE, 1, 8, FALSE),
(28, 2, 2, 4200.00, 1, 'Sales', 2, 55.0, 1, 2.5, 3.0, 2.0, 25.0, 0, FALSE, 2, 12, TRUE),
(42, 12, 1, 12000.00, 4, 'Engineering', 6, 45.0, 4, 4.5, 4.8, 4.5, 8.0, 3, TRUE, 1, 15, FALSE),
(25, 1, 1, 3500.00, 1, 'Marketing', 1, 52.0, 0, 2.0, 2.5, 1.5, 35.0, 0, FALSE, 3, 6, TRUE),
(38, 10, 3, 9500.00, 3, 'Product', 5, 44.0, 3, 4.0, 4.3, 3.8, 15.0, 2, TRUE, 1, 10, FALSE),
(45, 15, 2, 14500.00, 5, 'Engineering', 7, 48.0, 5, 4.7, 4.9, 4.2, 5.0, 4, TRUE, 0, 20, FALSE),
(26, 1, 1, 3800.00, 1, 'Sales', 2, 58.0, 1, 2.2, 2.8, 1.8, 40.0, 0, FALSE, 2, 8, TRUE),
(40, 11, 1, 11000.00, 4, 'Product', 6, 46.0, 4, 4.4, 4.6, 4.3, 10.0, 3, TRUE, 1, 12, FALSE),
(30, 3, 2, 5500.00, 2, 'Marketing', 3, 50.0, 2, 3.0, 3.5, 2.5, 20.0, 1, FALSE, 2, 7, FALSE),
(48, 18, 2, 16000.00, 5, 'Engineering', 8, 50.0, 6, 4.8, 5.0, 4.5, 3.0, 4, TRUE, 0, 25, FALSE),
(27, 1, 1, 3600.00, 1, 'Support', 2, 56.0, 0, 2.3, 2.6, 1.6, 38.0, 0, FALSE, 3, 10, TRUE),
(36, 9, 2, 8800.00, 3, 'Product', 5, 43.0, 3, 4.1, 4.4, 3.9, 14.0, 2, TRUE, 1, 9, FALSE),
(43, 14, 1, 13000.00, 4, 'Engineering', 7, 47.0, 5, 4.6, 4.7, 4.4, 6.0, 3, TRUE, 1, 18, FALSE),
(24, 1, 1, 3400.00, 1, 'Marketing', 1, 54.0, 1, 2.1, 2.4, 1.7, 32.0, 0, FALSE, 2, 5, TRUE),
(39, 10, 3, 9800.00, 3, 'Sales', 4, 44.5, 3, 4.0, 4.2, 3.7, 16.0, 2, TRUE, 1, 11, FALSE),
(41, 13, 2, 12500.00, 4, 'Product', 6, 46.5, 4, 4.5, 4.6, 4.1, 9.0, 3, TRUE, 1, 14, FALSE),
(29, 2, 2, 4500.00, 1, 'Support', 2, 53.0, 1, 2.6, 3.1, 2.1, 28.0, 0, FALSE, 2, 9, TRUE),
(46, 16, 2, 15000.00, 5, 'Engineering', 8, 49.0, 5, 4.7, 4.8, 4.4, 4.0, 4, TRUE, 0, 22, FALSE),
(32, 5, 3, 6500.00, 2, 'Marketing', 3, 48.0, 2, 3.2, 3.6, 2.8, 18.0, 1, FALSE, 2, 8, FALSE),
(44, 14, 1, 13500.00, 4, 'Product', 7, 47.5, 4, 4.6, 4.7, 4.3, 7.0, 3, TRUE, 1, 16, FALSE),
(26, 1, 1, 3700.00, 1, 'Sales', 2, 57.0, 1, 2.4, 2.9, 1.9, 36.0, 0, FALSE, 2, 7, TRUE),
(37, 9, 2, 9000.00, 3, 'Engineering', 5, 43.5, 3, 4.2, 4.4, 4.0, 13.0, 2, TRUE, 1, 10, FALSE),
(47, 17, 2, 15500.00, 5, 'Product', 8, 49.5, 6, 4.8, 4.9, 4.5, 2.0, 4, TRUE, 0, 24, FALSE),
(28, 2, 2, 4300.00, 1, 'Marketing', 2, 54.5, 1, 2.5, 3.0, 2.0, 30.0, 0, FALSE, 2, 6, TRUE),
(40, 12, 2, 10500.00, 3, 'Engineering', 5, 44.0, 4, 4.3, 4.5, 4.1, 11.0, 3, TRUE, 1, 13, FALSE);
