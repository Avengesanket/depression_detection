import os
import psycopg2
import streamlit as st
import datetime
import json

@st.cache_resource
def init_connection():
    """Initialize and cache the database connection."""
    try:
        return psycopg2.connect(
            host=st.secrets.database.host,
            port=st.secrets.database.port,
            database=st.secrets.database.dbname,
            user=st.secrets.database.user,
            password=st.secrets.database.password
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def get_db_connection():
    """Get the active database connection, re-establishing it if closed."""
    conn = init_connection()
    if conn is None:
        return None
    try:
       
        if conn.closed or conn.status != psycopg2.extensions.STATUS_READY:
            init_connection.clear()
            conn = init_connection()
    except psycopg2.OperationalError:
        init_connection.clear()
        conn = init_connection()

    return conn

def init_db():
    """Create required tables if they do not exist."""
    conn = get_db_connection()
    if not conn:
        st.error("Database not available. Cannot initialize tables.")
        return

    try:
        with conn.cursor() as cur:
            # Users table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    full_name VARCHAR(100),
                    age INTEGER,
                    gender VARCHAR(10),
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                );
            ''')

            # PHQ-9 responses table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS phq9_responses (
                    id SERIAL PRIMARY KEY,
                    user_id INT REFERENCES users(id) ON DELETE CASCADE,
                    question_index INT CHECK (question_index BETWEEN 0 AND 8),
                    response_value INT CHECK (response_value BETWEEN 0 AND 3),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, question_index)
                );
            ''')

            # Extended assessments table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS extended_assessments (
                    id SERIAL PRIMARY KEY,
                    user_id INT REFERENCES users(id) ON DELETE CASCADE UNIQUE,
                    assessment_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            # Suggestions table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_suggestions (
                    id SERIAL PRIMARY KEY,
                    user_id INT REFERENCES users(id) ON DELETE CASCADE UNIQUE,
                    suggestions_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Error initializing database: {e}")

def get_user_profile(username):
    """Retrieve user's age and gender by username."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT age, gender FROM users WHERE username = %s;",
                (username,)
            )
            result = cur.fetchone()
            if result:
                age, gender = result
                return {"age": age, "gender": gender}
            return None
    except Exception as e:
        st.error(f"Error retrieving user profile: {e}")
        return None

def save_phq9_response(user_id, question_index, response_value):
    """Insert or update a single PHQ-9 response."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection is closed or unavailable.")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO phq9_responses (user_id, question_index, response_value, timestamp)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, question_index)
                DO UPDATE SET response_value = EXCLUDED.response_value,
                              timestamp = EXCLUDED.timestamp;
            """, (user_id, question_index, response_value, datetime.datetime.now(datetime.timezone.utc)))
            conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Database error while saving PHQ-9 response: {e}")
        return False

def get_saved_phq9_responses(user_id):
    """
    Retrieve all saved PHQ-9 responses for a user, even if incomplete.
    Returns a dictionary of {question_index: response_value}.
    """
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question_index, response_value
                FROM phq9_responses
                WHERE user_id = %s
                ORDER BY question_index;
            """, (user_id,))
            results = cur.fetchall()
            
            saved_scores = {q_idx: value for q_idx, value in results}
            return saved_scores

    except Exception as e:
        st.error(f"Failed to load PHQ-9 responses: {e}")
        return None

def save_extra_assessment_data(user_id, data_dict):
    """Save or update extended assessment data as JSON."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection is closed or unavailable.")
        return False

    try:
        sanitized_data = json.dumps(data_dict)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extended_assessments (user_id, assessment_data)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET assessment_data = EXCLUDED.assessment_data,
                              timestamp = CURRENT_TIMESTAMP;
            """, (user_id, sanitized_data))
            conn.commit()
        return True
    except (psycopg2.DatabaseError, TypeError) as e:
        conn.rollback()
        st.error(f"Data save failed: {e}")
        return False

def get_extra_assessment_data(user_id):
    """Retrieve extended assessment data for a user."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection is closed or unavailable.")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT assessment_data, timestamp
                FROM extended_assessments
                WHERE user_id = %s;
            """, (user_id,))
            
            result = cur.fetchone()
            if result:
                return {
                    "assessment_data": result[0],
                    "timestamp": result[1]
                }
            return None
    except psycopg2.DatabaseError as e:
        st.error(f"Data retrieval failed: {e}")
        return None

def save_user_suggestions(user_id, report_dict):
    """Save or update the personalized suggestions report."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection is closed or unavailable.")
        return False

    try:
        payload = json.dumps(report_dict)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_suggestions (user_id, suggestions_data)
                VALUES (%s, %s)
                ON CONFLICT (user_id)
                  DO UPDATE SET suggestions_data = EXCLUDED.suggestions_data,
                                timestamp = CURRENT_TIMESTAMP;
            """, (user_id, payload))
            conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to save suggestions: {e}")
        return False
    
def get_user_suggestions(user_id):
    """Retrieve the saved suggestions report for a user."""
    conn = get_db_connection()
    if not conn:
        st.error("Database connection is closed or unavailable.")
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT suggestions_data, timestamp
                FROM user_suggestions
                WHERE user_id = %s;
            """, (user_id,))
            
            result = cur.fetchone()
            if result:
                return {
                    "suggestions_data": result[0],
                    "timestamp": result[1]
                }
            return None
    except psycopg2.DatabaseError as e:
        st.error(f"Suggestion retrieval failed: {e}")
        return None

def delete_user_data_by_table(table_name, user_id):
    """Generic function to delete user data from a specific table."""
    conn = get_db_connection()
    if not conn:
        st.error("Database not available for data deletion.")
        return False

    allowed_tables = ["phq9_responses", "extended_assessments", "user_suggestions"]
    if table_name not in allowed_tables:
        st.error(f"Deletion from table '{table_name}' is not permitted.")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {table_name} WHERE user_id = %s;", (user_id,))
            conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Failed to delete data from {table_name}: {e}")
        return False

def delete_phq9_responses(user_id):
    """Deletes all PHQ-9 responses for a user."""
    return delete_user_data_by_table("phq9_responses", user_id)

def delete_extra_assessment_data(user_id):
    """Deletes the extended assessment data for a user."""
    return delete_user_data_by_table("extended_assessments", user_id)

def delete_user_suggestions(user_id):
    """Deletes the suggestions data for a user."""
    return delete_user_data_by_table("user_suggestions", user_id)