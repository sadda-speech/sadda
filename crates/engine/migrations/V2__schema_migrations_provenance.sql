-- V2: extend schema_migrations with name + checksum columns.
--
-- The migration runner writes a row into schema_migrations for each applied
-- migration with all four columns set. For DBs that pre-date V2 (a single
-- Phase-0 row with NULL name/checksum) the migrator's V2 closure backfills
-- the V1 row from on-disk SQL in the same transaction as these ALTER TABLEs.

ALTER TABLE schema_migrations ADD COLUMN name     TEXT;
ALTER TABLE schema_migrations ADD COLUMN checksum TEXT;
