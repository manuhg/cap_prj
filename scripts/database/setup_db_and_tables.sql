-- Database: PostgreSQL Database
-- - Tables:
--     - conversations:
--         - title: string (max 256 chars)
--         - id: conversation_id (128 chars hash. Primary key and indexed)
--         - created_at: timestamp (indexed with descending order)
--         - status: string (fixed - 10 chars) [PINNED, HIDDEN, ARCHIVED] (foreign key to conversation_statuses table) 
--     - conversations_log:
--         - created_at: timestamp (indexed with descending order)
--         - conversation_id: string (hash) (fixed_len: 128 chars, indexed)
--         - s_no: auto increment bigint
--         - text: string (variable length)
--         - s_no and conversation id joint primary key
--

create database tldr;
\c tldr;
-- Table to store allowed conversation statuses
CREATE TABLE conversation_statuses (
                                       status VARCHAR(10) PRIMARY KEY
);

-- Insert predefined statuses
INSERT INTO conversation_statuses (status) VALUES ('PINNED'), ('HIDDEN'), ('ARCHIVED');

-- Conversations table
CREATE TABLE conversations (
                               id CHAR(128) PRIMARY KEY,  -- Fixed length hash as primary key
                               title VARCHAR(256) NOT NULL,
                               created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                               status VARCHAR(10) NOT NULL REFERENCES conversation_statuses(status)
);

-- Index for efficient sorting by created_at in descending order
CREATE INDEX idx_conversations_created_at ON conversations (created_at DESC);

-- Conversations log table
CREATE TABLE conversations_log (
                                   conversation_id CHAR(128) NOT NULL,
                                   s_no BIGSERIAL NOT NULL,  -- Auto-incrementing serial number
                                   created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                                   text TEXT NOT NULL,

                                   PRIMARY KEY (conversation_id, s_no),
                                   FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Index for quick lookup by conversation_id
CREATE INDEX idx_conversations_log_conversation_id ON conversations_log (conversation_id);

-- Index for sorting logs by creation time in descending order
CREATE INDEX idx_conversations_log_created_at ON conversations_log (created_at DESC);