"""
Initialize database schemas for PostgreSQL and Neo4j.

This script:
1. Initializes PostgreSQL schema from schema.sql
2. Initializes Neo4j schema from neo4j_schema.cypher

WARNING: This will create the database schemas if they don't exist!
"""

import asyncio
import sys
from pathlib import Path
from psqlpy import ConnectionPool

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    print("❌ neo4j driver not installed. Install with: pip install neo4j")
    sys.exit(1)

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mem.config import get_config


async def init_postgresql():
    """Initialize PostgreSQL schema."""
    config = get_config()

    # Build DSN
    dsn = (
        f"postgresql://{config.postgres_user}:{config.postgres_password}"
        f"@{config.postgres_host}:{config.postgres_port}/{config.postgres_db}"
    )

    print(f"🔗 Connecting to PostgreSQL: {config.postgres_db}")
    pool = ConnectionPool(dsn=dsn, max_db_pool_size=2)

    try:
        conn = await pool.connection()

        print("\n🏗️  Initializing PostgreSQL schema...")

        # Ensure extensions exist
        print("\n🔧 Ensuring extensions...")
        extensions = [
            ("vector", "Vector similarity search"),
            ("vchord", "Hierarchical vector search"),
            ("pg_tokenizer", "Text tokenization"),
            ("vchord_bm25", "BM25 full-text search"),
        ]
        for ext_name, desc in extensions:
            try:
                await conn.execute(f"CREATE EXTENSION IF NOT EXISTS {ext_name} CASCADE")
                print(f"   ✅ {ext_name}: {desc}")
            except Exception as e:
                print(f"   ⚠️  {ext_name}: {e}")

        # Read and execute schema.sql
        print("\n📄 Reading schema.sql...")
        schema_path = Path(__file__).parent.parent / "sql" / "schema.sql"

        if not schema_path.exists():
            print(f"   ❌ Schema file not found: {schema_path}")
            return

        schema_sql = schema_path.read_text(encoding="utf-8")

        # Split into statements and execute
        # Remove comments and split by semicolons, but not inside dollar-quoted strings
        statements = []
        current = []
        in_dollar = False

        for line in schema_sql.split("\n"):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("--"):
                continue

            # Track dollar-quoted strings
            dollar_count = line.count("$$")
            for _ in range(dollar_count):
                in_dollar = not in_dollar

            current.append(line)

            # Split on semicolon only if not inside dollar quotes
            if ";" in line and not in_dollar:
                statements.append("\n".join(current))
                current = []

        if current:
            statements.append("\n".join(current))

        # Execute each statement
        for i, statement in enumerate(statements, 1):
            statement = statement.strip()
            if not statement:
                continue

            try:
                await conn.execute(statement)
                # Extract table/object name for logging
                if "CREATE TABLE" in statement:
                    table_name = statement.split("CREATE TABLE")[1].split("(")[0].strip().split()[0]
                    print(f"   ✅ Table: {table_name}")
                elif "CREATE INDEX" in statement:
                    index_name = (
                        statement.split("CREATE INDEX")[1].split("ON")[0].strip().split()[0]
                    )
                    print(f"   ✅ Index: {index_name}")
                elif "CREATE TRIGGER" in statement:
                    trigger_name = statement.split("CREATE TRIGGER")[1].split("BEFORE")[0].strip()
                    print(f"   ✅ Trigger: {trigger_name}")
                elif "CREATE OR REPLACE FUNCTION" in statement or "CREATE FUNCTION" in statement:
                    func_name = statement.split("FUNCTION")[1].split("(")[0].strip()
                    print(f"   ✅ Function: {func_name}")
                elif "SELECT create_tokenizer" in statement:
                    print(f"   ✅ Tokenizer: bert")
                elif "CREATE OR REPLACE VIEW" in statement or "CREATE VIEW" in statement:
                    view_name = statement.split("VIEW")[1].split("AS")[0].strip()
                    print(f"   ✅ View: {view_name}")
            except Exception as e:
                print(f"   ❌ Error in statement {i}: {e}")
                print(f"      Statement: {statement[:100]}...")

        print("\n✅ PostgreSQL schema initialization complete!")

    finally:
        pool.close()


async def init_neo4j():
    """Initialize Neo4j schema."""
    config = get_config()

    # Use Neo4j URI directly
    uri = config.neo4j_uri
    auth = (config.neo4j_user, config.neo4j_password)

    print(f"🔗 Connecting to Neo4j: {config.neo4j_user}@{uri}")

    driver = AsyncGraphDatabase.driver(uri, auth=auth)

    try:
        async with driver.session() as session:
            print("\n🏗️  Initializing Neo4j schema...")

            # Read neo4j_schema.cypher
            schema_path = Path(__file__).parent.parent / "sql" / "neo4j_schema.cypher"

            if not schema_path.exists():
                print(f"   ❌ Neo4j schema file not found: {schema_path}")
                return

            schema_cypher = schema_path.read_text(encoding="utf-8")

            # Split into statements (Cypher uses semicolons like SQL)
            statements = []
            current = []
            in_comment = False

            for line in schema_cypher.split("\n"):
                stripped = line.strip()

                # Handle multi-line comments
                if stripped.startswith("/*"):
                    in_comment = True
                if in_comment:
                    if "*/" in line:
                        in_comment = False
                    continue

                # Skip single-line comments and empty lines
                if not stripped or stripped.startswith("//"):
                    continue

                current.append(line)

                # Split on semicolon
                if ";" in line:
                    statements.append("\n".join(current))
                    current = []

            if current:
                statements.append("\n".join(current))

            # Execute each statement
            for i, statement in enumerate(statements, 1):
                statement = statement.strip()
                if not statement:
                    continue

                try:
                    await session.run(statement)
                    # Extract operation for logging
                    if "CREATE INDEX" in statement.upper():
                        if "IF NOT EXISTS" in statement:
                            try:
                                index_part = (
                                    statement.split("IF NOT EXISTS")[1].split("FOR")[0].strip()
                                )
                                print(f"   ✅ Index: {index_part}")
                            except:
                                print(f"   ✅ Index created")
                        else:
                            print(f"   ✅ Index created")
                    elif "CREATE CONSTRAINT" in statement.upper():
                        if "IF NOT EXISTS" in statement:
                            try:
                                constraint_part = (
                                    statement.split("IF NOT EXISTS")[1].split("FOR")[0].strip()
                                )
                                print(f"   ✅ Constraint: {constraint_part}")
                            except:
                                print(f"   ✅ Constraint created")
                        else:
                            print(f"   ✅ Constraint created")
                    else:
                        print(f"   ✅ Statement executed")
                except Exception as e:
                    # Check if it's an "already exists" error, which is OK
                    error_msg = str(e).lower()
                    if (
                        "already exists" in error_msg
                        or "indexalreadyexists" in error_msg
                        or "constraintalreadyexists" in error_msg
                    ):
                        print(f"   ℹ️  Already exists (skipping)")
                    else:
                        print(f"   ❌ Error in statement {i}: {e}")
                        print(f"      Statement: {statement[:100]}...")

        print("\n✅ Neo4j schema initialization complete!")

    finally:
        await driver.close()


async def main():
    """Initialize both databases."""
    print("=" * 70)
    print("🚀 Initializing AgentMem Databases")
    print("=" * 70)

    try:
        await init_postgresql()
        print()
        await init_neo4j()

        print("\n" + "=" * 70)
        print("✅ All database schemas initialized successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
