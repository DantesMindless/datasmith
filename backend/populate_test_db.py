"""
Populate a test PostgreSQL database with blood test data.
This connects to a separate test database and creates tables with realistic medical data.
"""

import psycopg2
from psycopg2 import sql
import numpy as np
from datetime import datetime, timedelta

# Test database configuration
# UPDATE THESE WITH YOUR TEST DATABASE CREDENTIALS
DB_CONFIG = {
    'host': 'localhost',  # or your DB host
    'port': 5434,         # or your DB port
    'database': 'postgres',  # your test database name
    'user': 'postgres',   # your database user
    'password': 'postgres'  # your database password
}

def create_connection():
    """Create database connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"Connected to database: {DB_CONFIG['database']}")
        return conn
    except Exception as e:
        print(f" Error connecting to database: {e}")
        raise

def create_tables(conn):
    """Create blood test tables."""
    cursor = conn.cursor()

    # Drop existing tables
    print("\nDropping existing tables if they exist...")
    cursor.execute("DROP TABLE IF EXISTS blood_tests CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS patients CASCADE;")

    # Create patients table
    print("Creating patients table...")
    cursor.execute("""
        CREATE TABLE patients (
            patient_id VARCHAR(10) PRIMARY KEY,
            age INTEGER NOT NULL,
            gender VARCHAR(1) NOT NULL CHECK (gender IN ('M', 'F')),
            bmi DECIMAL(4, 1) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create blood_tests table
    print("Creating blood_tests table...")
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
            disease_risk VARCHAR(10) NOT NULL CHECK (disease_risk IN ('High', 'Low')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create indexes
    print("Creating indexes...")
    cursor.execute("CREATE INDEX idx_patient_id ON blood_tests(patient_id);")
    cursor.execute("CREATE INDEX idx_test_date ON blood_tests(test_date);")
    cursor.execute("CREATE INDEX idx_disease_risk ON blood_tests(disease_risk);")

    conn.commit()
    print(" Tables created successfully!\n")

def generate_and_insert_data(conn, num_patients=10000):
    """Generate and insert blood test data."""
    cursor = conn.cursor()

    print(f"Generating {num_patients} patient records with blood test data...")

    np.random.seed(42)

    # Prepare batch insert
    patient_records = []
    blood_test_records = []

    for i in range(1, num_patients + 1):
        patient_id = f'P{str(i).zfill(6)}'
        age = int(np.random.randint(18, 81))
        gender = np.random.choice(['M', 'F'])
        bmi = round(float(np.random.normal(26, 5)), 1)

        # Patient data
        patient_records.append((patient_id, age, gender, bmi))

        # Blood test data
        hemoglobin = round(float(np.random.normal(15, 2)), 2)
        wbc_count = round(float(np.random.normal(7.5, 2)), 2)
        platelet_count = int(np.random.normal(275, 75))
        rbc_count = round(float(np.random.normal(5.2, 0.5)), 2)
        glucose = round(float(np.random.normal(100, 25)), 1)
        cholesterol_total = round(float(np.random.normal(180, 40)), 1)
        hdl_cholesterol = round(float(np.random.normal(55, 15)), 1)
        ldl_cholesterol = round(float(np.random.normal(110, 35)), 1)
        triglycerides = round(float(np.random.normal(120, 50)), 1)
        bp_systolic = int(np.random.normal(125, 20))
        bp_diastolic = int(np.random.normal(80, 12))

        # Generate test date within last year
        days_ago = np.random.randint(0, 365)
        test_date = (datetime.now() - timedelta(days=days_ago)).date()

        # Calculate disease risk
        risk_score = 0
        if age > 60:
            risk_score += 2
        elif age > 45:
            risk_score += 1
        if hemoglobin < 12 or hemoglobin > 17:
            risk_score += 1
        if wbc_count < 4 or wbc_count > 11:
            risk_score += 1
        if glucose > 125:
            risk_score += 2
        elif glucose > 100:
            risk_score += 1
        if cholesterol_total > 200:
            risk_score += 1
        if hdl_cholesterol < 40:
            risk_score += 1
        if ldl_cholesterol > 130:
            risk_score += 1
        if triglycerides > 150:
            risk_score += 1
        if bp_systolic > 140 or bp_diastolic > 90:
            risk_score += 2
        elif bp_systolic > 130 or bp_diastolic > 85:
            risk_score += 1
        if bmi > 30:
            risk_score += 2
        elif bmi > 25:
            risk_score += 1

        disease_risk = 'High' if risk_score >= 6 else 'Low'

        # Add 5% random noise
        if np.random.random() < 0.05:
            disease_risk = 'Low' if disease_risk == 'High' else 'High'

        blood_test_records.append((
            patient_id, test_date, hemoglobin, wbc_count, platelet_count,
            rbc_count, glucose, cholesterol_total, hdl_cholesterol,
            ldl_cholesterol, triglycerides, bp_systolic, bp_diastolic,
            disease_risk
        ))

        # Progress indicator
        if i % 1000 == 0:
            print(f"  Generated {i}/{num_patients} records...")

    # Batch insert patients
    print("\nInserting patient records...")
    cursor.executemany(
        "INSERT INTO patients (patient_id, age, gender, bmi) VALUES (%s, %s, %s, %s)",
        patient_records
    )

    # Batch insert blood tests
    print("Inserting blood test records...")
    cursor.executemany("""
        INSERT INTO blood_tests (
            patient_id, test_date, hemoglobin, wbc_count, platelet_count,
            rbc_count, glucose, cholesterol_total, hdl_cholesterol,
            ldl_cholesterol, triglycerides, bp_systolic, bp_diastolic,
            disease_risk
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, blood_test_records)

    conn.commit()
    print(f" Inserted {len(patient_records)} patients and {len(blood_test_records)} blood tests!\n")

def show_statistics(conn):
    """Display database statistics."""
    cursor = conn.cursor()

    print("=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    # Patient statistics
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]
    print(f"\nTotal Patients: {total_patients}")

    cursor.execute("SELECT gender, COUNT(*) FROM patients GROUP BY gender ORDER BY gender")
    print("\nGender Distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]}")

    cursor.execute("SELECT MIN(age), MAX(age), ROUND(AVG(age), 1) FROM patients")
    min_age, max_age, avg_age = cursor.fetchone()
    print(f"\nAge Range: {min_age} - {max_age} (avg: {avg_age})")

    # Blood test statistics
    cursor.execute("SELECT COUNT(*) FROM blood_tests")
    total_tests = cursor.fetchone()[0]
    print(f"\nTotal Blood Tests: {total_tests}")

    cursor.execute("SELECT disease_risk, COUNT(*) FROM blood_tests GROUP BY disease_risk ORDER BY disease_risk")
    print("\nDisease Risk Distribution:")
    for row in cursor.fetchall():
        percentage = (row[1] / total_tests) * 100
        print(f"  {row[0]}: {row[1]} ({percentage:.1f}%)")

    # Sample data
    cursor.execute("""
        SELECT p.patient_id, p.age, p.gender, bt.glucose, bt.cholesterol_total,
               bt.bp_systolic, bt.bp_diastolic, bt.disease_risk
        FROM patients p
        JOIN blood_tests bt ON p.patient_id = bt.patient_id
        LIMIT 10
    """)

    print("\nSample Data (first 10 records):")
    print("-" * 100)
    print(f"{'Patient ID':<12} {'Age':<5} {'Gender':<7} {'Glucose':<9} {'Cholesterol':<12} {'BP':<12} {'Risk':<6}")
    print("-" * 100)
    for row in cursor.fetchall():
        bp = f"{row[5]}/{row[6]}"
        print(f"{row[0]:<12} {row[1]:<5} {row[2]:<7} {row[3]:<9.1f} {row[4]:<12.1f} {bp:<12} {row[7]:<6}")

    print("=" * 60)

def main():
    """Main execution."""
    print("=" * 60)
    print("TEST DATABASE POPULATION SCRIPT")
    print("Blood Test Data Generator")
    print("=" * 60)
    print(f"\nTarget Database: {DB_CONFIG['database']}")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print("\n" + "=" * 60)

    # Confirm before proceeding
    response = input("\nWARNING: This will DROP existing tables and create new ones. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user.")
        return

    try:
        # Connect to database
        conn = create_connection()

        # Create tables
        create_tables(conn)

        # Generate and insert data
        num_patients = 10000
        generate_and_insert_data(conn, num_patients)

        # Show statistics
        show_statistics(conn)

        # Close connection
        conn.close()
        print("\n Database population complete!")
        print("\nYou can now query the following tables:")
        print("  - patients (patient demographics)")
        print("  - blood_tests (blood test results with disease risk)")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
