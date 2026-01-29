"""
Populate a test PostgreSQL database with diverse datasets for ML training.
Creates multiple realistic scenarios:
1. Healthcare - Blood tests & disease risk prediction
2. E-commerce - Customer churn prediction
3. Finance - Loan default prediction
4. HR - Employee attrition prediction
5. Manufacturing - Quality defect prediction
6. Real Estate - House price prediction (regression)
7. Marketing - Campaign conversion prediction
"""

import psycopg2
from psycopg2 import sql
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Test database configuration
DB_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'postgres'
}

def create_connection():
    """Create database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"Connected to database: {DB_CONFIG['database']}")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise


# =============================================================================
# 1. HEALTHCARE - Blood Tests & Disease Risk
# =============================================================================
def create_healthcare_tables(conn):
    """Create healthcare/blood test tables."""
    cursor = conn.cursor()

    print("\n[HEALTHCARE] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS blood_tests CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS patients CASCADE;")

    cursor.execute("""
        CREATE TABLE patients (
            patient_id VARCHAR(10) PRIMARY KEY,
            age INTEGER NOT NULL,
            gender VARCHAR(1) NOT NULL CHECK (gender IN ('M', 'F')),
            bmi DECIMAL(4, 1) NOT NULL,
            smoking_status VARCHAR(20) DEFAULT 'never',
            family_history BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE blood_tests (
            test_id SERIAL PRIMARY KEY,
            patient_id VARCHAR(10) NOT NULL REFERENCES patients(patient_id),
            test_date DATE NOT NULL,
            hemoglobin DECIMAL(4, 2) NOT NULL,
            wbc_count DECIMAL(4, 2) NOT NULL,
            platelet_count INTEGER NOT NULL,
            rbc_count DECIMAL(3, 2) NOT NULL,
            glucose DECIMAL(5, 1) NOT NULL,
            cholesterol_total DECIMAL(5, 1) NOT NULL,
            hdl_cholesterol DECIMAL(4, 1) NOT NULL,
            ldl_cholesterol DECIMAL(5, 1) NOT NULL,
            triglycerides DECIMAL(5, 1) NOT NULL,
            bp_systolic INTEGER NOT NULL,
            bp_diastolic INTEGER NOT NULL,
            heart_rate INTEGER NOT NULL,
            disease_risk VARCHAR(10) NOT NULL CHECK (disease_risk IN ('High', 'Low')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("CREATE INDEX idx_bt_patient_id ON blood_tests(patient_id);")
    cursor.execute("CREATE INDEX idx_bt_disease_risk ON blood_tests(disease_risk);")
    conn.commit()
    print("[HEALTHCARE] Tables created!")

def populate_healthcare_data(conn, num_patients=10000):
    """Generate healthcare data."""
    cursor = conn.cursor()
    print(f"[HEALTHCARE] Generating {num_patients} patient records...")

    np.random.seed(42)
    patient_records = []
    blood_test_records = []

    smoking_options = ['never', 'former', 'current']

    for i in range(1, num_patients + 1):
        patient_id = f'P{str(i).zfill(6)}'
        age = int(np.random.randint(18, 85))
        gender = np.random.choice(['M', 'F'])
        bmi = round(float(np.clip(np.random.normal(26, 5), 16, 45)), 1)
        smoking = np.random.choice(smoking_options, p=[0.5, 0.3, 0.2])
        family_history = np.random.random() < 0.25

        patient_records.append((patient_id, age, gender, bmi, smoking, family_history))

        # Generate blood test with realistic correlations
        hemoglobin = round(float(np.clip(np.random.normal(14 if gender == 'M' else 13, 1.5), 8, 20)), 2)
        wbc_count = round(float(np.clip(np.random.normal(7.5, 2), 2, 15)), 2)
        platelet_count = int(np.clip(np.random.normal(250, 60), 100, 450))
        rbc_count = round(float(np.clip(np.random.normal(5.0, 0.5), 3.5, 6.5)), 2)
        glucose = round(float(np.clip(np.random.normal(95 + age * 0.3, 25), 60, 300)), 1)
        cholesterol_total = round(float(np.clip(np.random.normal(180 + age * 0.5, 35), 100, 350)), 1)
        hdl_cholesterol = round(float(np.clip(np.random.normal(55, 12), 25, 100)), 1)
        ldl_cholesterol = round(float(np.clip(cholesterol_total - hdl_cholesterol - 30, 50, 250)), 1)
        triglycerides = round(float(np.clip(np.random.normal(120, 45), 50, 400)), 1)
        bp_systolic = int(np.clip(np.random.normal(120 + age * 0.3, 15), 90, 200))
        bp_diastolic = int(np.clip(np.random.normal(75 + age * 0.1, 10), 55, 120))
        heart_rate = int(np.clip(np.random.normal(72, 12), 50, 120))

        days_ago = np.random.randint(0, 365)
        test_date = (datetime.now() - timedelta(days=days_ago)).date()

        # Calculate disease risk with logical rules
        risk_score = 0
        if age > 60: risk_score += 2
        elif age > 45: risk_score += 1
        if hemoglobin < 11 or hemoglobin > 17: risk_score += 1
        if glucose > 140: risk_score += 3
        elif glucose > 110: risk_score += 1
        if cholesterol_total > 240: risk_score += 2
        elif cholesterol_total > 200: risk_score += 1
        if hdl_cholesterol < 40: risk_score += 2
        if bp_systolic > 140 or bp_diastolic > 90: risk_score += 2
        if bmi > 30: risk_score += 2
        elif bmi > 27: risk_score += 1
        if smoking == 'current': risk_score += 2
        elif smoking == 'former': risk_score += 1
        if family_history: risk_score += 1

        disease_risk = 'High' if risk_score >= 6 else 'Low'
        if np.random.random() < 0.03:  # 3% noise
            disease_risk = 'Low' if disease_risk == 'High' else 'High'

        blood_test_records.append((
            patient_id, test_date, hemoglobin, wbc_count, platelet_count,
            rbc_count, glucose, cholesterol_total, hdl_cholesterol,
            ldl_cholesterol, triglycerides, bp_systolic, bp_diastolic,
            heart_rate, disease_risk
        ))

    cursor.executemany(
        "INSERT INTO patients (patient_id, age, gender, bmi, smoking_status, family_history) VALUES (%s, %s, %s, %s, %s, %s)",
        patient_records
    )
    cursor.executemany("""
        INSERT INTO blood_tests (patient_id, test_date, hemoglobin, wbc_count, platelet_count,
            rbc_count, glucose, cholesterol_total, hdl_cholesterol, ldl_cholesterol,
            triglycerides, bp_systolic, bp_diastolic, heart_rate, disease_risk)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, blood_test_records)

    conn.commit()
    print(f"[HEALTHCARE] Inserted {len(patient_records)} patients and blood tests!")


# =============================================================================
# 2. E-COMMERCE - Customer Churn Prediction
# =============================================================================
def create_ecommerce_tables(conn):
    """Create e-commerce customer churn tables."""
    cursor = conn.cursor()

    print("\n[E-COMMERCE] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS customer_orders CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS customers CASCADE;")

    cursor.execute("""
        CREATE TABLE customers (
            customer_id VARCHAR(12) PRIMARY KEY,
            signup_date DATE NOT NULL,
            age INTEGER,
            gender VARCHAR(10),
            location_tier VARCHAR(10),
            preferred_category VARCHAR(30),
            account_type VARCHAR(20) DEFAULT 'standard',
            total_orders INTEGER DEFAULT 0,
            total_spent DECIMAL(10, 2) DEFAULT 0,
            avg_order_value DECIMAL(8, 2) DEFAULT 0,
            days_since_last_order INTEGER,
            support_tickets INTEGER DEFAULT 0,
            returns_count INTEGER DEFAULT 0,
            loyalty_points INTEGER DEFAULT 0,
            email_opt_in BOOLEAN DEFAULT TRUE,
            app_installed BOOLEAN DEFAULT FALSE,
            churned BOOLEAN NOT NULL
        );
    """)

    cursor.execute("""
        CREATE TABLE customer_orders (
            order_id VARCHAR(15) PRIMARY KEY,
            customer_id VARCHAR(12) REFERENCES customers(customer_id),
            order_date DATE NOT NULL,
            order_value DECIMAL(8, 2) NOT NULL,
            items_count INTEGER NOT NULL,
            category VARCHAR(30),
            payment_method VARCHAR(20),
            discount_applied DECIMAL(5, 2) DEFAULT 0,
            delivery_days INTEGER
        );
    """)

    cursor.execute("CREATE INDEX idx_co_customer_id ON customer_orders(customer_id);")
    conn.commit()
    print("[E-COMMERCE] Tables created!")

def populate_ecommerce_data(conn, num_customers=8000):
    """Generate e-commerce churn data."""
    cursor = conn.cursor()
    print(f"[E-COMMERCE] Generating {num_customers} customer records...")

    np.random.seed(43)
    customer_records = []
    order_records = []

    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Grocery']
    locations = ['Tier1', 'Tier2', 'Tier3']
    account_types = ['standard', 'premium', 'vip']
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'cod']

    for i in range(1, num_customers + 1):
        customer_id = f'CUST{str(i).zfill(8)}'
        signup_days_ago = np.random.randint(30, 1095)  # 1 month to 3 years
        signup_date = (datetime.now() - timedelta(days=signup_days_ago)).date()

        age = int(np.clip(np.random.normal(35, 12), 18, 75))
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        location = np.random.choice(locations, p=[0.3, 0.45, 0.25])
        preferred_category = np.random.choice(categories)
        account_type = np.random.choice(account_types, p=[0.7, 0.25, 0.05])

        # Generate order history
        base_orders = np.random.poisson(8)
        if account_type == 'premium': base_orders = int(base_orders * 1.5)
        if account_type == 'vip': base_orders = int(base_orders * 2.5)
        total_orders = max(1, base_orders)

        avg_order_value = float(np.clip(np.random.normal(75, 40), 15, 500))
        if account_type == 'vip': avg_order_value *= 1.8
        total_spent = round(total_orders * avg_order_value * np.random.uniform(0.8, 1.2), 2)

        days_since_last = int(np.random.exponential(45))
        support_tickets = int(np.random.poisson(1))
        returns_count = int(np.random.poisson(0.5))
        loyalty_points = int(total_spent * 10 * np.random.uniform(0.5, 1.0))
        email_opt_in = np.random.random() > 0.15
        app_installed = np.random.random() > 0.6

        # Churn logic - realistic patterns
        churn_score = 0
        if days_since_last > 90: churn_score += 3
        elif days_since_last > 60: churn_score += 2
        elif days_since_last > 30: churn_score += 1
        if total_orders < 3: churn_score += 2
        if support_tickets > 2: churn_score += 2
        if returns_count > 2: churn_score += 1
        if not email_opt_in: churn_score += 1
        if not app_installed: churn_score += 1
        if account_type == 'standard': churn_score += 1
        if avg_order_value < 30: churn_score += 1

        churned = churn_score >= 5
        if np.random.random() < 0.05:  # 5% noise
            churned = not churned

        customer_records.append((
            customer_id, signup_date, age, gender, location, preferred_category,
            account_type, total_orders, total_spent, round(avg_order_value, 2),
            days_since_last, support_tickets, returns_count, loyalty_points,
            email_opt_in, app_installed, churned
        ))

        # Generate some orders
        for j in range(min(total_orders, 5)):
            order_id = f'ORD{str(i).zfill(6)}{str(j).zfill(3)}'
            order_days_ago = np.random.randint(max(days_since_last, 1), max(signup_days_ago, days_since_last + 1))
            order_date = (datetime.now() - timedelta(days=order_days_ago)).date()
            order_value = round(float(np.clip(np.random.normal(avg_order_value, 30), 10, 1000)), 2)
            items_count = int(np.clip(np.random.poisson(3), 1, 20))
            category = np.random.choice(categories)
            payment = np.random.choice(payment_methods, p=[0.4, 0.3, 0.2, 0.1])
            discount = round(float(np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05])), 2)
            delivery = int(np.clip(np.random.normal(4, 2), 1, 14))

            order_records.append((order_id, customer_id, order_date, order_value, items_count,
                                 category, payment, discount, delivery))

    cursor.executemany("""
        INSERT INTO customers (customer_id, signup_date, age, gender, location_tier,
            preferred_category, account_type, total_orders, total_spent, avg_order_value,
            days_since_last_order, support_tickets, returns_count, loyalty_points,
            email_opt_in, app_installed, churned)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, customer_records)

    cursor.executemany("""
        INSERT INTO customer_orders (order_id, customer_id, order_date, order_value,
            items_count, category, payment_method, discount_applied, delivery_days)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, order_records)

    conn.commit()
    print(f"[E-COMMERCE] Inserted {len(customer_records)} customers and {len(order_records)} orders!")


# =============================================================================
# 3. FINANCE - Loan Default Prediction
# =============================================================================
def create_finance_tables(conn):
    """Create finance/loan tables."""
    cursor = conn.cursor()

    print("\n[FINANCE] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS loan_payments CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS loans CASCADE;")

    cursor.execute("""
        CREATE TABLE loans (
            loan_id VARCHAR(12) PRIMARY KEY,
            applicant_age INTEGER NOT NULL,
            applicant_income DECIMAL(12, 2) NOT NULL,
            employment_length INTEGER,
            employment_type VARCHAR(20),
            home_ownership VARCHAR(20),
            loan_amount DECIMAL(12, 2) NOT NULL,
            loan_term INTEGER NOT NULL,
            interest_rate DECIMAL(5, 2) NOT NULL,
            loan_purpose VARCHAR(30),
            credit_score INTEGER NOT NULL,
            dti_ratio DECIMAL(5, 2),
            num_credit_lines INTEGER,
            num_delinquencies INTEGER DEFAULT 0,
            total_credit_limit DECIMAL(12, 2),
            credit_utilization DECIMAL(5, 2),
            months_since_delinquency INTEGER,
            application_date DATE NOT NULL,
            defaulted BOOLEAN NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX idx_loans_defaulted ON loans(defaulted);")
    cursor.execute("CREATE INDEX idx_loans_credit_score ON loans(credit_score);")
    conn.commit()
    print("[FINANCE] Tables created!")

def populate_finance_data(conn, num_loans=12000):
    """Generate loan default data."""
    cursor = conn.cursor()
    print(f"[FINANCE] Generating {num_loans} loan records...")

    np.random.seed(44)
    loan_records = []

    employment_types = ['full_time', 'part_time', 'self_employed', 'contract', 'unemployed']
    home_types = ['rent', 'own', 'mortgage', 'other']
    purposes = ['debt_consolidation', 'home_improvement', 'business', 'education', 'medical', 'car', 'vacation', 'other']

    for i in range(1, num_loans + 1):
        loan_id = f'LN{str(i).zfill(10)}'

        age = int(np.clip(np.random.normal(38, 12), 21, 70))
        employment_type = np.random.choice(employment_types, p=[0.6, 0.1, 0.15, 0.1, 0.05])

        # Income based on employment
        base_income = 55000
        if employment_type == 'self_employed': base_income = 70000
        elif employment_type == 'part_time': base_income = 28000
        elif employment_type == 'unemployed': base_income = 15000
        income = float(np.clip(np.random.normal(base_income, base_income * 0.4), 12000, 500000))

        employment_length = int(np.clip(np.random.exponential(5), 0, 30)) if employment_type != 'unemployed' else 0
        home_ownership = np.random.choice(home_types, p=[0.35, 0.25, 0.35, 0.05])

        loan_amount = float(np.clip(np.random.exponential(15000), 1000, min(income * 3, 100000)))
        loan_term = np.random.choice([12, 24, 36, 48, 60, 72], p=[0.1, 0.15, 0.35, 0.2, 0.15, 0.05])

        # Credit score with realistic distribution
        credit_score = int(np.clip(np.random.normal(680, 80), 300, 850))

        # Interest rate based on credit score
        base_rate = 18 - (credit_score - 300) * 0.02
        interest_rate = round(float(np.clip(base_rate + np.random.normal(0, 2), 4, 30)), 2)

        purpose = np.random.choice(purposes, p=[0.35, 0.15, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

        dti_ratio = round(float(np.clip((loan_amount / 12) / (income / 12) * 100, 5, 60)), 2)
        num_credit_lines = int(np.clip(np.random.poisson(5), 1, 25))
        num_delinquencies = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]))

        total_credit_limit = float(np.clip(income * np.random.uniform(0.5, 2), 5000, 200000))
        credit_utilization = round(float(np.clip(np.random.beta(2, 5) * 100, 0, 100)), 2)

        months_since_delinquency = None if num_delinquencies == 0 else int(np.random.exponential(24))

        days_ago = np.random.randint(30, 730)
        application_date = (datetime.now() - timedelta(days=days_ago)).date()

        # Default logic
        default_score = 0
        if credit_score < 600: default_score += 4
        elif credit_score < 650: default_score += 2
        elif credit_score < 700: default_score += 1
        if dti_ratio > 40: default_score += 3
        elif dti_ratio > 30: default_score += 1
        if num_delinquencies > 2: default_score += 3
        elif num_delinquencies > 0: default_score += 1
        if employment_type == 'unemployed': default_score += 3
        if credit_utilization > 80: default_score += 2
        elif credit_utilization > 50: default_score += 1
        if loan_amount > income * 0.5: default_score += 2
        if interest_rate > 20: default_score += 1

        defaulted = default_score >= 6
        if np.random.random() < 0.04:  # 4% noise
            defaulted = not defaulted

        loan_records.append((
            loan_id, int(age), round(income, 2), int(employment_length), employment_type,
            home_ownership, round(loan_amount, 2), int(loan_term), interest_rate, purpose,
            int(credit_score), dti_ratio, int(num_credit_lines), int(num_delinquencies),
            round(total_credit_limit, 2), credit_utilization, int(months_since_delinquency) if months_since_delinquency else None,
            application_date, defaulted
        ))

    cursor.executemany("""
        INSERT INTO loans (loan_id, applicant_age, applicant_income, employment_length,
            employment_type, home_ownership, loan_amount, loan_term, interest_rate,
            loan_purpose, credit_score, dti_ratio, num_credit_lines, num_delinquencies,
            total_credit_limit, credit_utilization, months_since_delinquency,
            application_date, defaulted)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, loan_records)

    conn.commit()
    print(f"[FINANCE] Inserted {len(loan_records)} loan records!")


# =============================================================================
# 4. HR - Employee Attrition Prediction
# =============================================================================
def create_hr_tables(conn):
    """Create HR/employee tables."""
    cursor = conn.cursor()

    print("\n[HR] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS employees CASCADE;")

    cursor.execute("""
        CREATE TABLE employees (
            employee_id VARCHAR(10) PRIMARY KEY,
            age INTEGER NOT NULL,
            gender VARCHAR(10),
            education_level VARCHAR(20),
            department VARCHAR(30) NOT NULL,
            job_role VARCHAR(40) NOT NULL,
            job_level INTEGER NOT NULL,
            years_at_company INTEGER NOT NULL,
            years_in_role INTEGER NOT NULL,
            years_since_promotion INTEGER,
            total_working_years INTEGER NOT NULL,
            num_companies_worked INTEGER,
            monthly_income DECIMAL(10, 2) NOT NULL,
            percent_salary_hike INTEGER,
            performance_rating INTEGER,
            training_times_last_year INTEGER,
            work_life_balance INTEGER,
            job_satisfaction INTEGER,
            environment_satisfaction INTEGER,
            relationship_satisfaction INTEGER,
            overtime VARCHAR(3),
            business_travel VARCHAR(20),
            distance_from_home INTEGER,
            stock_option_level INTEGER,
            attrition BOOLEAN NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX idx_emp_attrition ON employees(attrition);")
    cursor.execute("CREATE INDEX idx_emp_department ON employees(department);")
    conn.commit()
    print("[HR] Tables created!")

def populate_hr_data(conn, num_employees=5000):
    """Generate employee attrition data."""
    cursor = conn.cursor()
    print(f"[HR] Generating {num_employees} employee records...")

    np.random.seed(45)
    employee_records = []

    departments = ['Sales', 'Engineering', 'Marketing', 'HR', 'Finance', 'Operations', 'R&D', 'Support']
    job_roles = {
        'Sales': ['Sales Rep', 'Sales Manager', 'Account Executive', 'Sales Director'],
        'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
        'Marketing': ['Marketing Analyst', 'Marketing Manager', 'Content Specialist', 'CMO'],
        'HR': ['HR Coordinator', 'HR Manager', 'Recruiter', 'HR Director'],
        'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'CFO'],
        'Operations': ['Operations Analyst', 'Operations Manager', 'Supply Chain Manager'],
        'R&D': ['Research Scientist', 'Lab Technician', 'R&D Manager'],
        'Support': ['Support Agent', 'Support Lead', 'Support Manager']
    }
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    travel_types = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']

    for i in range(1, num_employees + 1):
        employee_id = f'EMP{str(i).zfill(6)}'

        age = int(np.clip(np.random.normal(36, 10), 22, 60))
        gender = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        education = np.random.choice(education_levels, p=[0.15, 0.45, 0.32, 0.08])

        department = np.random.choice(departments)
        job_role = np.random.choice(job_roles[department])
        job_level = int(np.clip(np.random.poisson(2) + 1, 1, 5))

        total_working_years = max(0, age - 22 - np.random.randint(0, 5))
        years_at_company = int(np.clip(np.random.exponential(4), 0, min(total_working_years, 25)))
        years_in_role = int(np.clip(np.random.exponential(2), 0, years_at_company))
        years_since_promotion = int(np.clip(np.random.exponential(2), 0, years_at_company))
        num_companies = max(1, total_working_years // 4 + np.random.randint(-1, 2))

        # Income based on job level and experience
        base_income = 3000 + job_level * 2000 + total_working_years * 200
        monthly_income = round(float(np.clip(np.random.normal(base_income, base_income * 0.2), 2000, 20000)), 2)

        percent_salary_hike = int(np.clip(np.random.normal(13, 4), 0, 25))
        performance_rating = np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.60, 0.20])
        training_times = int(np.clip(np.random.poisson(2), 0, 6))

        work_life_balance = np.random.choice([1, 2, 3, 4], p=[0.1, 0.25, 0.45, 0.2])
        job_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.12, 0.22, 0.40, 0.26])
        env_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.45, 0.25])
        rel_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.08, 0.2, 0.45, 0.27])

        overtime = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        business_travel = np.random.choice(travel_types, p=[0.15, 0.70, 0.15])
        distance_from_home = int(np.clip(np.random.exponential(10), 1, 50))
        stock_option_level = np.random.choice([0, 1, 2, 3], p=[0.45, 0.35, 0.15, 0.05])

        # Attrition logic
        attrition_score = 0
        if job_satisfaction <= 2: attrition_score += 2
        if env_satisfaction <= 2: attrition_score += 1
        if work_life_balance <= 2: attrition_score += 2
        if overtime == 'Yes': attrition_score += 2
        if years_since_promotion > 5: attrition_score += 2
        if monthly_income < 4000: attrition_score += 2
        if business_travel == 'Travel_Frequently': attrition_score += 1
        if distance_from_home > 20: attrition_score += 1
        if years_at_company < 2: attrition_score += 1
        if stock_option_level == 0: attrition_score += 1
        if age < 30: attrition_score += 1

        attrition = attrition_score >= 6
        if np.random.random() < 0.05:
            attrition = not attrition

        employee_records.append((
            employee_id, age, gender, education, department, job_role, job_level,
            years_at_company, years_in_role, years_since_promotion, total_working_years,
            num_companies, monthly_income, percent_salary_hike, performance_rating,
            training_times, work_life_balance, job_satisfaction, env_satisfaction,
            rel_satisfaction, overtime, business_travel, distance_from_home,
            stock_option_level, attrition
        ))

    cursor.executemany("""
        INSERT INTO employees (employee_id, age, gender, education_level, department,
            job_role, job_level, years_at_company, years_in_role, years_since_promotion,
            total_working_years, num_companies_worked, monthly_income, percent_salary_hike,
            performance_rating, training_times_last_year, work_life_balance, job_satisfaction,
            environment_satisfaction, relationship_satisfaction, overtime, business_travel,
            distance_from_home, stock_option_level, attrition)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, employee_records)

    conn.commit()
    print(f"[HR] Inserted {len(employee_records)} employee records!")


# =============================================================================
# 5. MANUFACTURING - Quality Defect Prediction
# =============================================================================
def create_manufacturing_tables(conn):
    """Create manufacturing/quality tables."""
    cursor = conn.cursor()

    print("\n[MANUFACTURING] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS production_batches CASCADE;")

    cursor.execute("""
        CREATE TABLE production_batches (
            batch_id VARCHAR(15) PRIMARY KEY,
            production_date DATE NOT NULL,
            production_line VARCHAR(10) NOT NULL,
            shift VARCHAR(10) NOT NULL,
            operator_experience INTEGER,
            machine_age_months INTEGER NOT NULL,
            temperature DECIMAL(5, 2) NOT NULL,
            humidity DECIMAL(5, 2) NOT NULL,
            pressure DECIMAL(6, 2) NOT NULL,
            vibration_level DECIMAL(5, 2) NOT NULL,
            speed_rpm INTEGER NOT NULL,
            material_quality_score DECIMAL(4, 2) NOT NULL,
            maintenance_days_ago INTEGER,
            batch_size INTEGER NOT NULL,
            processing_time_minutes INTEGER NOT NULL,
            ambient_temp DECIMAL(5, 2),
            power_consumption DECIMAL(8, 2),
            defect_rate DECIMAL(5, 2) NOT NULL,
            has_defects BOOLEAN NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX idx_pb_has_defects ON production_batches(has_defects);")
    cursor.execute("CREATE INDEX idx_pb_production_line ON production_batches(production_line);")
    conn.commit()
    print("[MANUFACTURING] Tables created!")

def populate_manufacturing_data(conn, num_batches=15000):
    """Generate manufacturing quality data."""
    cursor = conn.cursor()
    print(f"[MANUFACTURING] Generating {num_batches} production batch records...")

    np.random.seed(46)
    batch_records = []

    production_lines = ['LINE_A', 'LINE_B', 'LINE_C', 'LINE_D']
    shifts = ['Morning', 'Afternoon', 'Night']

    for i in range(1, num_batches + 1):
        batch_id = f'BATCH{str(i).zfill(9)}'

        days_ago = np.random.randint(0, 365)
        production_date = (datetime.now() - timedelta(days=days_ago)).date()

        production_line = np.random.choice(production_lines)
        shift = np.random.choice(shifts, p=[0.4, 0.4, 0.2])

        operator_experience = int(np.clip(np.random.exponential(24), 1, 120))  # months
        machine_age_months = int(np.clip(np.random.exponential(36), 1, 120))

        # Process parameters with realistic ranges
        temperature = round(float(np.clip(np.random.normal(180, 15), 140, 220)), 2)
        humidity = round(float(np.clip(np.random.normal(45, 10), 20, 80)), 2)
        pressure = round(float(np.clip(np.random.normal(1013, 50), 900, 1100)), 2)
        vibration = round(float(np.clip(np.random.exponential(2), 0.5, 10)), 2)
        speed_rpm = int(np.clip(np.random.normal(1500, 200), 1000, 2000))

        material_quality = round(float(np.clip(np.random.normal(8.5, 1), 5, 10)), 2)
        maintenance_days_ago = int(np.clip(np.random.exponential(15), 1, 90))
        batch_size = int(np.clip(np.random.normal(500, 100), 200, 1000))
        processing_time = int(np.clip(np.random.normal(45, 10), 20, 90))
        ambient_temp = round(float(np.clip(np.random.normal(22, 3), 15, 35)), 2)
        power_consumption = round(float(np.clip(np.random.normal(150, 30), 80, 250)), 2)

        # Defect logic based on parameters
        defect_prob = 0.05  # base defect rate

        if temperature < 160 or temperature > 200: defect_prob += 0.15
        if humidity > 60 or humidity < 30: defect_prob += 0.10
        if vibration > 5: defect_prob += 0.12
        if machine_age_months > 60: defect_prob += 0.08
        if maintenance_days_ago > 30: defect_prob += 0.10
        if material_quality < 7: defect_prob += 0.15
        if operator_experience < 6: defect_prob += 0.08
        if shift == 'Night': defect_prob += 0.05
        if speed_rpm > 1700 or speed_rpm < 1300: defect_prob += 0.08

        defect_rate = round(float(np.clip(defect_prob * 100 + np.random.normal(0, 2), 0, 50)), 2)
        has_defects = defect_rate > 5  # More than 5% defect rate

        batch_records.append((
            batch_id, production_date, production_line, shift, operator_experience,
            machine_age_months, temperature, humidity, pressure, vibration,
            speed_rpm, material_quality, maintenance_days_ago, batch_size,
            processing_time, ambient_temp, power_consumption, defect_rate, has_defects
        ))

    cursor.executemany("""
        INSERT INTO production_batches (batch_id, production_date, production_line, shift,
            operator_experience, machine_age_months, temperature, humidity, pressure,
            vibration_level, speed_rpm, material_quality_score, maintenance_days_ago,
            batch_size, processing_time_minutes, ambient_temp, power_consumption,
            defect_rate, has_defects)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, batch_records)

    conn.commit()
    print(f"[MANUFACTURING] Inserted {len(batch_records)} production batch records!")


# =============================================================================
# 6. REAL ESTATE - House Price Prediction (Regression)
# =============================================================================
def create_realestate_tables(conn):
    """Create real estate tables."""
    cursor = conn.cursor()

    print("\n[REAL ESTATE] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS houses CASCADE;")

    cursor.execute("""
        CREATE TABLE houses (
            property_id VARCHAR(12) PRIMARY KEY,
            listing_date DATE NOT NULL,
            neighborhood VARCHAR(30) NOT NULL,
            property_type VARCHAR(20) NOT NULL,
            year_built INTEGER NOT NULL,
            lot_size_sqft INTEGER NOT NULL,
            living_area_sqft INTEGER NOT NULL,
            bedrooms INTEGER NOT NULL,
            bathrooms DECIMAL(3, 1) NOT NULL,
            floors INTEGER NOT NULL,
            garage_spaces INTEGER,
            has_pool BOOLEAN DEFAULT FALSE,
            has_fireplace BOOLEAN DEFAULT FALSE,
            has_basement BOOLEAN DEFAULT FALSE,
            condition_rating INTEGER NOT NULL,
            renovation_year INTEGER,
            distance_to_downtown DECIMAL(5, 2),
            distance_to_school DECIMAL(5, 2),
            crime_rate_area DECIMAL(5, 2),
            avg_income_area INTEGER,
            price DECIMAL(12, 2) NOT NULL,
            price_category VARCHAR(10) NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX idx_houses_neighborhood ON houses(neighborhood);")
    cursor.execute("CREATE INDEX idx_houses_price_cat ON houses(price_category);")
    conn.commit()
    print("[REAL ESTATE] Tables created!")

def populate_realestate_data(conn, num_houses=10000):
    """Generate real estate data."""
    cursor = conn.cursor()
    print(f"[REAL ESTATE] Generating {num_houses} property records...")

    np.random.seed(47)
    house_records = []

    neighborhoods = ['Downtown', 'Midtown', 'Uptown', 'Suburbs North', 'Suburbs South',
                    'Suburbs East', 'Suburbs West', 'Lakefront', 'Historic District', 'Tech Park']
    property_types = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family']

    # Neighborhood price multipliers
    neighborhood_multipliers = {
        'Downtown': 1.4, 'Midtown': 1.2, 'Uptown': 1.3, 'Lakefront': 1.5,
        'Historic District': 1.25, 'Tech Park': 1.15, 'Suburbs North': 1.0,
        'Suburbs South': 0.9, 'Suburbs East': 0.95, 'Suburbs West': 0.92
    }

    for i in range(1, num_houses + 1):
        property_id = f'PROP{str(i).zfill(8)}'

        days_ago = np.random.randint(0, 730)
        listing_date = (datetime.now() - timedelta(days=days_ago)).date()

        neighborhood = np.random.choice(neighborhoods)
        property_type = np.random.choice(property_types, p=[0.5, 0.2, 0.2, 0.1])

        year_built = int(np.clip(np.random.normal(1985, 25), 1900, 2024))

        # Size based on property type
        if property_type == 'Single Family':
            lot_size = int(np.clip(np.random.normal(8000, 3000), 3000, 25000))
            living_area = int(np.clip(np.random.normal(2000, 600), 1000, 5000))
            bedrooms = int(np.clip(np.random.normal(3.5, 1), 2, 6))
        elif property_type == 'Townhouse':
            lot_size = int(np.clip(np.random.normal(2500, 800), 1500, 5000))
            living_area = int(np.clip(np.random.normal(1600, 400), 1000, 3000))
            bedrooms = int(np.clip(np.random.normal(3, 0.8), 2, 4))
        elif property_type == 'Condo':
            lot_size = int(np.clip(np.random.normal(1000, 300), 500, 2000))
            living_area = int(np.clip(np.random.normal(1200, 400), 600, 2500))
            bedrooms = int(np.clip(np.random.normal(2, 0.7), 1, 4))
        else:  # Multi-Family
            lot_size = int(np.clip(np.random.normal(6000, 2000), 3000, 15000))
            living_area = int(np.clip(np.random.normal(3000, 800), 2000, 6000))
            bedrooms = int(np.clip(np.random.normal(5, 1.5), 3, 10))

        bathrooms = round(float(np.clip(bedrooms * 0.75 + np.random.normal(0, 0.5), 1, 6)), 1)
        floors = np.random.choice([1, 2, 3], p=[0.3, 0.55, 0.15])
        garage_spaces = np.random.choice([0, 1, 2, 3], p=[0.15, 0.35, 0.40, 0.10])

        has_pool = np.random.random() < 0.15 and property_type in ['Single Family', 'Multi-Family']
        has_fireplace = np.random.random() < 0.40
        has_basement = np.random.random() < 0.50 and property_type != 'Condo'

        condition = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.35, 0.35, 0.10])
        renovation_year = year_built + np.random.randint(10, 40) if np.random.random() < 0.3 else None
        if renovation_year and renovation_year > 2024: renovation_year = None

        distance_downtown = round(float(np.clip(np.random.exponential(8), 0.5, 30)), 2)
        distance_school = round(float(np.clip(np.random.exponential(2), 0.2, 10)), 2)
        crime_rate = round(float(np.clip(np.random.exponential(3), 0.5, 15)), 2)
        avg_income = int(np.clip(np.random.normal(75000, 25000), 30000, 200000))

        # Calculate price with realistic factors
        base_price = living_area * 150  # $150 per sqft base

        price = base_price
        price *= neighborhood_multipliers[neighborhood]
        price *= (1 + (condition - 3) * 0.08)  # condition adjustment
        price *= (1 - (2024 - year_built) * 0.003)  # age depreciation
        if renovation_year and renovation_year > 2010: price *= 1.10
        if has_pool: price *= 1.08
        if has_fireplace: price *= 1.03
        if has_basement: price *= 1.05
        price *= (1 + garage_spaces * 0.03)
        price *= (1 - distance_downtown * 0.01)
        price *= (1 - crime_rate * 0.02)
        price *= (1 + (avg_income - 75000) / 75000 * 0.15)

        # Add some randomness
        price *= np.random.uniform(0.9, 1.1)
        price = round(max(50000, price), 2)

        # Categorize price
        if price < 200000: price_category = 'Budget'
        elif price < 400000: price_category = 'Mid-Range'
        elif price < 700000: price_category = 'Premium'
        else: price_category = 'Luxury'

        house_records.append((
            property_id, listing_date, neighborhood, property_type, year_built,
            lot_size, living_area, bedrooms, bathrooms, floors, garage_spaces,
            has_pool, has_fireplace, has_basement, condition, renovation_year,
            distance_downtown, distance_school, crime_rate, avg_income, price, price_category
        ))

    cursor.executemany("""
        INSERT INTO houses (property_id, listing_date, neighborhood, property_type,
            year_built, lot_size_sqft, living_area_sqft, bedrooms, bathrooms, floors,
            garage_spaces, has_pool, has_fireplace, has_basement, condition_rating,
            renovation_year, distance_to_downtown, distance_to_school, crime_rate_area,
            avg_income_area, price, price_category)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, house_records)

    conn.commit()
    print(f"[REAL ESTATE] Inserted {len(house_records)} property records!")


# =============================================================================
# 7. MARKETING - Campaign Conversion Prediction
# =============================================================================
def create_marketing_tables(conn):
    """Create marketing campaign tables."""
    cursor = conn.cursor()

    print("\n[MARKETING] Creating tables...")
    cursor.execute("DROP TABLE IF EXISTS campaign_responses CASCADE;")

    cursor.execute("""
        CREATE TABLE campaign_responses (
            response_id VARCHAR(12) PRIMARY KEY,
            campaign_date DATE NOT NULL,
            campaign_type VARCHAR(20) NOT NULL,
            channel VARCHAR(20) NOT NULL,
            customer_age INTEGER,
            customer_income_bracket VARCHAR(15),
            customer_education VARCHAR(20),
            customer_marital_status VARCHAR(15),
            previous_purchases INTEGER,
            days_since_last_purchase INTEGER,
            email_open_rate DECIMAL(5, 2),
            click_rate DECIMAL(5, 2),
            website_visits_30d INTEGER,
            time_on_site_minutes DECIMAL(6, 2),
            pages_viewed INTEGER,
            discount_offered DECIMAL(5, 2),
            personalization_score DECIMAL(4, 2),
            send_hour INTEGER,
            send_day_of_week VARCHAR(10),
            device_type VARCHAR(15),
            converted BOOLEAN NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX idx_cr_converted ON campaign_responses(converted);")
    cursor.execute("CREATE INDEX idx_cr_campaign_type ON campaign_responses(campaign_type);")
    conn.commit()
    print("[MARKETING] Tables created!")

def populate_marketing_data(conn, num_responses=20000):
    """Generate marketing campaign data."""
    cursor = conn.cursor()
    print(f"[MARKETING] Generating {num_responses} campaign response records...")

    np.random.seed(48)
    response_records = []

    campaign_types = ['Email', 'SMS', 'Push', 'Social', 'Display']
    channels = ['Direct', 'Organic', 'Paid Search', 'Social Media', 'Referral', 'Affiliate']
    income_brackets = ['<30k', '30k-50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
    education_levels = ['High School', 'Some College', 'Bachelor', 'Master', 'PhD']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    devices = ['Desktop', 'Mobile', 'Tablet']

    for i in range(1, num_responses + 1):
        response_id = f'RESP{str(i).zfill(8)}'

        days_ago = np.random.randint(0, 180)
        campaign_date = (datetime.now() - timedelta(days=days_ago)).date()

        campaign_type = np.random.choice(campaign_types, p=[0.4, 0.15, 0.15, 0.2, 0.1])
        channel = np.random.choice(channels, p=[0.15, 0.2, 0.25, 0.2, 0.12, 0.08])

        age = int(np.clip(np.random.normal(38, 14), 18, 75))
        income_bracket = np.random.choice(income_brackets, p=[0.15, 0.25, 0.25, 0.18, 0.12, 0.05])
        education = np.random.choice(education_levels, p=[0.2, 0.25, 0.35, 0.15, 0.05])
        marital_status = np.random.choice(marital_statuses, p=[0.35, 0.45, 0.15, 0.05])

        previous_purchases = int(np.clip(np.random.exponential(3), 0, 30))
        days_since_last = int(np.clip(np.random.exponential(30), 1, 365)) if previous_purchases > 0 else 999

        email_open_rate = round(float(np.clip(np.random.beta(2, 5) * 100, 0, 100)), 2)
        click_rate = round(float(np.clip(np.random.beta(1, 8) * 100, 0, 50)), 2)
        website_visits = int(np.clip(np.random.exponential(5), 0, 50))
        time_on_site = round(float(np.clip(np.random.exponential(8), 0, 60)), 2)
        pages_viewed = int(np.clip(np.random.exponential(4), 1, 30))

        discount_offered = round(float(np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.3, 0.2, 0.2, 0.12, 0.1, 0.05, 0.03])), 2)
        personalization_score = round(float(np.clip(np.random.normal(6, 2), 1, 10)), 2)

        send_hour = int(np.random.choice(range(6, 22), p=np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.10, 0.12, 0.10, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.02])))
        send_day = np.random.choice(days_of_week, p=[0.12, 0.16, 0.16, 0.16, 0.15, 0.13, 0.12])
        device = np.random.choice(devices, p=[0.35, 0.55, 0.10])

        # Conversion logic
        conversion_prob = 0.03  # base conversion rate

        if previous_purchases > 5: conversion_prob += 0.08
        elif previous_purchases > 2: conversion_prob += 0.04
        if days_since_last < 30: conversion_prob += 0.05
        if email_open_rate > 30: conversion_prob += 0.04
        if click_rate > 10: conversion_prob += 0.06
        if website_visits > 10: conversion_prob += 0.04
        if time_on_site > 15: conversion_prob += 0.03
        if discount_offered >= 20: conversion_prob += 0.08
        elif discount_offered >= 10: conversion_prob += 0.04
        if personalization_score > 7: conversion_prob += 0.05
        if campaign_type == 'Email': conversion_prob += 0.02
        if send_hour in [10, 11, 14, 15]: conversion_prob += 0.02
        if send_day in ['Tuesday', 'Wednesday', 'Thursday']: conversion_prob += 0.01
        if income_bracket in ['75k-100k', '100k-150k', '>150k']: conversion_prob += 0.03

        converted = np.random.random() < conversion_prob

        response_records.append((
            response_id, campaign_date, campaign_type, channel, age, income_bracket,
            education, marital_status, previous_purchases, days_since_last,
            email_open_rate, click_rate, website_visits, time_on_site, pages_viewed,
            discount_offered, personalization_score, send_hour, send_day, device, converted
        ))

    cursor.executemany("""
        INSERT INTO campaign_responses (response_id, campaign_date, campaign_type, channel,
            customer_age, customer_income_bracket, customer_education, customer_marital_status,
            previous_purchases, days_since_last_purchase, email_open_rate, click_rate,
            website_visits_30d, time_on_site_minutes, pages_viewed, discount_offered,
            personalization_score, send_hour, send_day_of_week, device_type, converted)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, response_records)

    conn.commit()
    print(f"[MARKETING] Inserted {len(response_records)} campaign response records!")


# =============================================================================
# STATISTICS & SUMMARY
# =============================================================================
def show_all_statistics(conn):
    """Display statistics for all tables."""
    cursor = conn.cursor()

    print("\n" + "=" * 80)
    print("DATABASE POPULATION COMPLETE - SUMMARY")
    print("=" * 80)

    tables_info = [
        ('patients', 'Healthcare', 'patient_id'),
        ('blood_tests', 'Healthcare', 'disease_risk'),
        ('customers', 'E-Commerce', 'churned'),
        ('customer_orders', 'E-Commerce', 'order_id'),
        ('loans', 'Finance', 'defaulted'),
        ('employees', 'HR', 'attrition'),
        ('production_batches', 'Manufacturing', 'has_defects'),
        ('houses', 'Real Estate', 'price_category'),
        ('campaign_responses', 'Marketing', 'converted'),
    ]

    print(f"\n{'Table':<25} {'Domain':<15} {'Records':<12} {'Target Distribution'}")
    print("-" * 80)

    for table, domain, target_col in tables_info:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            if target_col in ['churned', 'defaulted', 'attrition', 'has_defects', 'converted']:
                cursor.execute(f"SELECT {target_col}, COUNT(*) FROM {table} GROUP BY {target_col}")
                dist = dict(cursor.fetchall())
                true_pct = dist.get(True, 0) / count * 100 if count > 0 else 0
                dist_str = f"True: {true_pct:.1f}%"
            elif target_col == 'disease_risk':
                cursor.execute(f"SELECT {target_col}, COUNT(*) FROM {table} GROUP BY {target_col}")
                dist = dict(cursor.fetchall())
                high_pct = dist.get('High', 0) / count * 100 if count > 0 else 0
                dist_str = f"High: {high_pct:.1f}%"
            elif target_col == 'price_category':
                cursor.execute(f"SELECT {target_col}, COUNT(*) FROM {table} GROUP BY {target_col}")
                results = cursor.fetchall()
                dist_str = ', '.join([f"{r[0]}: {r[1]}" for r in results[:3]])
            else:
                dist_str = "-"

            print(f"{table:<25} {domain:<15} {count:<12,} {dist_str}")
        except Exception as e:
            print(f"{table:<25} {domain:<15} ERROR: {e}")

    print("\n" + "=" * 80)
    print("\nAVAILABLE ML SCENARIOS:")
    print("-" * 80)
    print("""
1. HEALTHCARE (Classification)
   - Tables: patients + blood_tests (JOIN on patient_id)
   - Target: disease_risk (High/Low)
   - Features: age, gender, bmi, blood metrics, vitals
   - Use case: Predict cardiovascular disease risk

2. E-COMMERCE (Classification)
   - Table: customers
   - Target: churned (True/False)
   - Features: order history, engagement, support tickets
   - Use case: Predict customer churn

3. FINANCE (Classification)
   - Table: loans
   - Target: defaulted (True/False)
   - Features: income, credit score, DTI, employment
   - Use case: Predict loan default risk

4. HR (Classification)
   - Table: employees
   - Target: attrition (True/False)
   - Features: satisfaction scores, tenure, compensation
   - Use case: Predict employee turnover

5. MANUFACTURING (Classification)
   - Table: production_batches
   - Target: has_defects (True/False)
   - Features: machine params, operator experience, conditions
   - Use case: Predict quality defects

6. REAL ESTATE (Classification/Regression)
   - Table: houses
   - Target: price_category OR price (for regression)
   - Features: location, size, amenities, condition
   - Use case: Predict house prices or price category

7. MARKETING (Classification)
   - Table: campaign_responses
   - Target: converted (True/False)
   - Features: customer behavior, campaign params, timing
   - Use case: Predict campaign conversion
""")
    print("=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution."""
    print("=" * 80)
    print("DATASMITH - TEST DATABASE POPULATION SCRIPT")
    print("Multi-Domain ML Dataset Generator")
    print("=" * 80)
    print(f"\nTarget Database: {DB_CONFIG['database']}")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print("\nThis will create 7 different ML scenarios with realistic data:")
    print("  1. Healthcare (10,000 patients)")
    print("  2. E-Commerce (8,000 customers)")
    print("  3. Finance (12,000 loans)")
    print("  4. HR (5,000 employees)")
    print("  5. Manufacturing (15,000 batches)")
    print("  6. Real Estate (10,000 properties)")
    print("  7. Marketing (20,000 responses)")
    print("\n" + "=" * 80)

    response = input("\nWARNING: This will DROP existing tables. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user.")
        return

    try:
        conn = create_connection()

        # 1. Healthcare
        create_healthcare_tables(conn)
        populate_healthcare_data(conn, num_patients=10000)

        # 2. E-Commerce
        create_ecommerce_tables(conn)
        populate_ecommerce_data(conn, num_customers=8000)

        # 3. Finance
        create_finance_tables(conn)
        populate_finance_data(conn, num_loans=12000)

        # 4. HR
        create_hr_tables(conn)
        populate_hr_data(conn, num_employees=5000)

        # 5. Manufacturing
        create_manufacturing_tables(conn)
        populate_manufacturing_data(conn, num_batches=15000)

        # 6. Real Estate
        create_realestate_tables(conn)
        populate_realestate_data(conn, num_houses=10000)

        # 7. Marketing
        create_marketing_tables(conn)
        populate_marketing_data(conn, num_responses=20000)

        # Show summary
        show_all_statistics(conn)

        conn.close()
        print("\nDatabase population complete!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
