CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_file text NOT NULL,
  chunk_index integer NOT NULL,
  token_start integer NOT NULL,
  token_end integer NOT NULL,
  content text NOT NULL,
  embedding vector(384) NOT NULL,
  UNIQUE (source_file, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
  ON documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS rag_meta (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  embedding_model text NOT NULL,
  chunk_size integer NOT NULL,
  overlap integer NOT NULL,
  indexed_at timestamptz NOT NULL DEFAULT now()
);

-- Fungsi SQL ini dapat menjadi pola RPC pgvector. Urutan jarak ASC memungkinkan
-- planner memakai indeks HNSW bila ukuran tabel dan perkiraan biaya mendukung.
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(384),
  match_count integer DEFAULT 3
)
RETURNS TABLE (
  id bigint,
  source_file text,
  chunk_index integer,
  content text,
  distance double precision
)
LANGUAGE sql STABLE
AS $$
  SELECT d.id, d.source_file, d.chunk_index, d.content,
         (d.embedding <=> query_embedding)::double precision AS distance
  FROM documents AS d
  ORDER BY d.embedding <=> query_embedding ASC
  LIMIT LEAST(GREATEST(match_count, 1), 20);
$$;
