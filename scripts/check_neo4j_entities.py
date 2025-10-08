"""Check what entities are in Neo4j."""

import asyncio
from agent_mem.config.settings import get_config
from agent_mem.database.neo4j_manager import Neo4jManager


async def check():
    config = get_config()
    neo4j = Neo4jManager(config)
    await neo4j.initialize()

    async with neo4j.session() as session:
        # Count all shortterm entities
        result = await session.run("MATCH (e:ShorttermEntity) RETURN count(e) as count")
        record = await result.single()
        print(f"\nTotal ShorttermEntity nodes: {record['count']}")

        # Get sample entities
        result2 = await session.run(
            """
            MATCH (e:ShorttermEntity) 
            RETURN e.name as name, e.external_id as external_id, 
                   e.shortterm_memory_id as shortterm_memory_id
            LIMIT 20
            """
        )
        records = []
        async for record in result2:
            records.append(record)
        print(f"\nSample entities: (found {len(records)} records)")
        for r in records:
            print(
                f"  - '{r['name']}' (external_id={r['external_id']}, shortterm_memory_id={r['shortterm_memory_id']})"
            )

        # Count longterm entities
        result3 = await session.run("MATCH (e:LongtermEntity) RETURN count(e) as count")
        record3 = await result3.single()
        print(f"\nTotal LongtermEntity nodes: {record3['count']}")

        # Get sample longterm entities
        result4 = await session.run(
            """
            MATCH (e:LongtermEntity) 
            RETURN e.name as name, e.external_id as external_id
            LIMIT 20
            """
        )
        records4 = []
        async for record in result4:
            records4.append(record)
        print(f"\nSample longterm entities: (found {len(records4)} records)")
        for r in records4:
            print(f"  - '{r['name']}' (external_id={r['external_id']})")

    await neo4j.close()


if __name__ == "__main__":
    asyncio.run(check())
