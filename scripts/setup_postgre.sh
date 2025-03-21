brew install postgresql@14;
brew install pgvector;
brew services start postgresql@14;


# test it out !
psql postgres;

\c database_name;
create extension vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(3));
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
SELECT * FROM items ORDER BY embedding <-> '[3,5,8]' LIMIT 5;
