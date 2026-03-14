"""
Entity Traversal Service

Traverses relationships between entities in the knowledge graph.
Can find connected entities and their relationships.
"""

import re
from sqlalchemy import text
from core.database import get_engine


def detect_entities(query: str) -> list:
    """Extract potential entity names from query."""
    # Look for capitalized words (potential entities)
    entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    return list(set(entities))


def find_related_entities(entity: str, max_depth: int = 2) -> dict:
    """
    Find related entities in the knowledge graph.

    Args:
        entity: The entity to find relations for
        max_depth: How many hops to traverse

    Returns:
        dict with entity and its relationships
    """
    engine = get_engine()
    results = {"entity": entity, "relationships": [], "related_entities": set()}

    with engine.connect() as conn:
        # Find tables that might contain entity relationships
        tables_result = conn.execute(
            text("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        )

        all_tables = [r[0] for r in tables_result.fetchall()]

        for table in all_tables:
            # Get columns for this table
            try:
                cols_result = conn.execute(
                    text(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table}'
                """)
                )
                columns = [c[0] for c in cols_result.fetchall()]

                # Look for entity-related columns (name, source, target, type)
                entity_cols = [
                    c
                    for c in columns
                    if any(
                        x in c.lower()
                        for x in ["name", "entity", "source", "target", "type"]
                    )
                ]

                for col in entity_cols:
                    # Search for the entity
                    search_result = conn.execute(
                        text(f"""
                        SELECT * FROM "{table}" 
                        WHERE LOWER("{col}") LIKE '%{entity.lower()}%'
                        LIMIT 10
                    """)
                    )

                    rows = search_result.fetchall()
                    if rows:
                        keys = search_result.keys()
                        for row in rows:
                            row_dict = dict(zip(keys, row))

                            # Find related values
                            for key, value in row_dict.items():
                                if (
                                    value
                                    and isinstance(value, str)
                                    and value.lower() != entity.lower()
                                ):
                                    results["related_entities"].add(str(value))

                            results["relationships"].append(
                                {"table": table, "data": row_dict}
                            )

            except Exception as e:
                continue

    results["related_entities"] = list(results["related_entities"])[:10]
    return results


def traverse_relationships(start_entity: str, relation_type: str = None) -> dict:
    """
    Traverse knowledge graph relationships.

    Example: traverse_relationships("Graphene")
    finds all entities related to Graphene.
    """
    # First, find direct relationships
    direct = find_related_entities(start_entity)

    # Then, find second-degree connections
    second_degree = []
    for related in direct.get("related_entities", [])[:5]:
        second = find_related_entities(related)
        second_degree.extend(second.get("relationships", []))

    return {
        "start_entity": start_entity,
        "direct_relationships": direct.get("relationships", []),
        "direct_related": direct.get("related_entities", []),
        "second_degree_relationships": second_degree[:10],
    }


def extract_and_traverse(query: str) -> dict:
    """Extract entities from query and traverse their relationships."""
    entities = detect_entities(query)

    if not entities:
        return {"entities_found": [], "relationships": []}

    all_results = []
    for entity in entities[:3]:  # Limit to 3 entities
        traversal = traverse_relationships(entity)
        if traversal["direct_relationships"] or traversal["direct_related"]:
            all_results.append(traversal)

    return {"entities_found": entities, "relationship_data": all_results}
