"""
AgroSage 2.0 — auth.py
=======================
SQLite authentication + user profile + query history + admin viewer.
DB file: agrosage_users.db  (auto-created next to app.py)

Tables:
  users   — id, fullname, username, email, location, password_hash, created_at
  queries — id, user_id, question, answer, asked_at
"""

import sqlite3
import hashlib
import os
from functools import wraps
from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session, g)

auth_bp = Blueprint('auth', __name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agrosage_users.db')


# ═══════════════════════════════════════════════════════════════════
# DB HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_db():
    """Per-request SQLite connection (stored in Flask g)."""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def get_db_direct():
    """Opens a fresh connection outside request context (for app.py helpers)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def close_db(e=None):
    db = g.pop('db', None)
    if db:
        db.close()


def init_db():
    """Create tables on first run."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname      TEXT    NOT NULL,
            username      TEXT    NOT NULL UNIQUE,
            email         TEXT    NOT NULL UNIQUE,
            location      TEXT    DEFAULT '',
            password_hash TEXT    NOT NULL,
            created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS queries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            question  TEXT    NOT NULL,
            answer    TEXT    NOT NULL,
            asked_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    print(f"✅ DB ready → {DB_PATH}")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def register_teardown(app):
    app.teardown_appcontext(close_db)


# ═══════════════════════════════════════════════════════════════════
# USER CRUD  (called from app.py)
# ═══════════════════════════════════════════════════════════════════

def get_user_by_id(user_id: int):
    conn = get_db_direct()
    row  = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return row


def get_all_users():
    """Return all users WITHOUT password hash for admin view."""
    conn = get_db_direct()
    rows = conn.execute(
        "SELECT id, fullname, username, email, location, created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def update_profile(user_id: int, fullname: str, location: str):
    conn = get_db_direct()
    conn.execute(
        "UPDATE users SET fullname=?, location=? WHERE id=?",
        (fullname, location, user_id)
    )
    conn.commit()
    conn.close()


def save_query(user_id: int, question: str, answer: str):
    """Store a completed Q&A pair for the logged-in user."""
    conn = get_db_direct()
    conn.execute(
        "INSERT INTO queries (user_id, question, answer) VALUES (?,?,?)",
        (user_id, question, answer)
    )
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════
# LOGIN-REQUIRED DECORATOR
# ═══════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please login to access this page.', 'error')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('user_id'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        fullname         = request.form.get('fullname', '').strip()
        username         = request.form.get('username', '').strip().lower()
        email            = request.form.get('email', '').strip().lower()
        location         = request.form.get('location', '').strip()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []
        if not fullname:                   errors.append('Full name is required.')
        if len(username) < 3:              errors.append('Username must be at least 3 characters.')
        if '@' not in email:               errors.append('Enter a valid email address.')
        if len(password) < 6:             errors.append('Password must be at least 6 characters.')
        if password != confirm_password:  errors.append('Passwords do not match.')

        if errors:
            for err in errors: flash(err, 'error')
            return render_template('signup.html')

        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (fullname,username,email,location,password_hash) VALUES (?,?,?,?,?)",
                (fullname, username, email, location, _hash(password))
            )
            db.commit()
            flash('✅ Account created! Please login.', 'success')
            return redirect(url_for('auth.login'))
        except sqlite3.IntegrityError as e:
            msg = str(e)
            if 'username' in msg: flash('Username already taken.', 'error')
            elif 'email'  in msg: flash('Email already registered.', 'error')
            else:                 flash('Signup failed. Try again.', 'error')

    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Enter both username and password.', 'error')
            return render_template('login.html')

        db   = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=? AND password_hash=?",
            (username, _hash(password))
        ).fetchone()

        if user:
            session.permanent  = True
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['fullname'] = user['fullname']
            flash(f'✅ Welcome back, {user["fullname"]}!', 'success')
            return redirect(request.args.get('next') or url_for('home'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    name = session.get('fullname', 'User')
    session.clear()
    flash(f'👋 Goodbye, {name}!', 'success')
    return redirect(url_for('home'))
