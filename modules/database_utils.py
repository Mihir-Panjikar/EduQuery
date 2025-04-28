import sqlite3
import logging
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)
DB_FILE = "chat_history.db"

# These functions are kept for backward compatibility


def save_chat(user_query: str, generated_answer: str, context_sources: Optional[Dict[str, Any]] = None):
    """Legacy function - redirects to the new conversation-based system."""
    # Import streamlit locally to avoid circular imports
    import streamlit as st

    # Get or create a default conversation
    conversation_id = None
    if hasattr(st, 'session_state') and hasattr(st.session_state, 'current_conversation_id'):
        conversation_id = st.session_state.current_conversation_id

    if not conversation_id:
        # Create a new conversation
        conversation_id = create_conversation(f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                              st.session_state.get('selected_subject') if hasattr(st.session_state, 'selected_subject') else None)
        if hasattr(st, 'session_state'):
            st.session_state.current_conversation_id = conversation_id

    # Save the messages
    save_message(conversation_id, 'user', user_query)
    save_message(conversation_id, 'assistant',
                 generated_answer, context_sources)

# Rest of the imports and functions remain the same


def initialize_database():
    """Creates the database and the necessary tables if they don't exist."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if old chat_history table exists, and if so, save its data
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'")
        old_table_exists = cursor.fetchone() is not None
        old_chats = []

        if old_table_exists:
            try:
                cursor.execute(
                    'SELECT * FROM chat_history ORDER BY timestamp ASC')
                old_chats = cursor.fetchall()
                # Rename the old table
                cursor.execute(
                    'ALTER TABLE chat_history RENAME TO chat_history_old')
                logger.info(
                    "Renamed old chat_history table to chat_history_old")
            except sqlite3.Error as e:
                logger.error(f"Error backing up old chat history: {e}")

        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                subject TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                context_sources TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        ''')

        # Migrate old data if it exists
        if old_table_exists and old_chats:
            try:
                # Group chats by timestamp (simplified migration - in real scenario we might use a more sophisticated grouping)
                # Create one conversation for all old chats
                conversation_id = str(uuid.uuid4())
                cursor.execute(
                    'INSERT INTO conversations (id, title, subject) VALUES (?, ?, ?)',
                    (conversation_id, "Migrated Conversation", None)
                )

                # Insert each old message into the new format
                for chat in old_chats:
                    # Old format: id, timestamp, user_query, generated_answer, context_sources
                    chat_id, timestamp, user_query, generated_answer, context_sources = chat

                    # Adding user message
                    cursor.execute('''
                        INSERT INTO messages (conversation_id, timestamp, role, content, context_sources)
                        VALUES (?, ?, ?, ?, NULL)
                    ''', (conversation_id, timestamp, 'user', user_query))

                    # Adding assistant message
                    cursor.execute('''
                        INSERT INTO messages (conversation_id, timestamp, role, content, context_sources)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (conversation_id, timestamp, 'assistant', generated_answer, context_sources))

                logger.info(
                    "Successfully migrated old chat data to new format")
            except sqlite3.Error as e:
                logger.error(f"Error during chat data migration: {e}")

        conn.commit()
        logger.info(
            f"Database '{DB_FILE}' initialized successfully with conversation schema.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()


def create_conversation(title: str, subject: Optional[str] = None) -> str:
    """Creates a new conversation and returns its ID."""
    conn = None
    try:
        conversation_id = str(uuid.uuid4())
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (id, title, subject) VALUES (?, ?, ?)',
            (conversation_id, title, subject)
        )
        conn.commit()
        logger.debug(f"Created conversation with ID {conversation_id}")
        return conversation_id
    except sqlite3.Error as e:
        logger.error(f"Database error while creating conversation: {e}")
        return ""
    finally:
        if conn:
            conn.close()


def save_message(conversation_id: str, role: str, content: str, context_sources: Optional[Dict[str, Any]] = None):
    """Saves a message to a conversation."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Updating conversation timestamp
        cursor.execute(
            'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )

        # Adding the message
        sources_json = json.dumps(
            context_sources) if context_sources is not None else None
        cursor.execute('''
            INSERT INTO messages (conversation_id, role, content, context_sources)
            VALUES (?, ?, ?, ?)
        ''', (conversation_id, role, content, sources_json))

        conn.commit()
        logger.debug(f"Message saved to conversation {conversation_id}")
    except sqlite3.Error as e:
        logger.error(f"Database error while saving message: {e}")
    finally:
        if conn:
            conn.close()


def fetch_conversations():
    """Fetches all conversations from the database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        cursor.execute('''
            SELECT c.*, 
                   (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY timestamp ASC LIMIT 1) as first_message
            FROM conversations c
            ORDER BY c.updated_at DESC
        ''')

        rows = cursor.fetchall()
        conversations = []

        for row in rows:
            conversation = {
                'id': row['id'],
                'title': row['title'],
                'subject': row['subject'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'preview': row['first_message'][:30] + "..." if row['first_message'] and len(row['first_message']) > 30 else row['first_message']
            }
            conversations.append(conversation)

        return conversations
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching conversations: {e}")
        return []
    finally:
        if conn:
            conn.close()


def fetch_conversation_messages(conversation_id: str):
    """Fetches all messages for a specific conversation."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM messages 
            WHERE conversation_id = ? 
            ORDER BY timestamp ASC
        ''', (conversation_id,))

        rows = cursor.fetchall()
        messages = []

        for row in rows:
            message = {
                'id': row['id'],
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp'],
                'context_sources': json.loads(row['context_sources']) if row['context_sources'] else None
            }
            messages.append(message)

        return messages
    except sqlite3.Error as e:
        logger.error(
            f"Database error while fetching conversation messages: {e}")
        return []
    finally:
        if conn:
            conn.close()


def update_conversation_title(conversation_id: str, new_title: str):
    """Updates the title of a conversation."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE conversations SET title = ? WHERE id = ?',
            (new_title, conversation_id)
        )
        conn.commit()
        logger.debug(f"Updated title for conversation {conversation_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error while updating conversation title: {e}")
        return False
    finally:
        if conn:
            conn.close()


def delete_conversation(conversation_id: str):
    """Deletes a conversation and all its messages."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM conversations WHERE id = ?',
                       (conversation_id,))
        conn.commit()
        logger.debug(f"Deleted conversation {conversation_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error while deleting conversation: {e}")
        return False
    finally:
        if conn:
            conn.close()


def delete_all_conversations():
    """Deletes all conversations and messages."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM conversations')
        conn.commit()
        logger.debug("All conversations deleted successfully.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error while deleting all conversations: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Keeping these functions for backward compatibility during transition


def fetch_chat_history():
    """Legacy function that fetches all messages across all conversations."""
    pass


def delete_chat_entry(chat_id: int):
    """Legacy function - no longer applicable."""
    pass


def delete_all_chats():
    """Legacy function - redirects to delete_all_conversations."""
    return delete_all_conversations()


def get_chat_entry(chat_id: int):
    """Legacy function - no longer applicable."""
    pass
