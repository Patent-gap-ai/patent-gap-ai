"""Try the memory layer on YOUR real MongoDB and ChromaDB.

    cd Backend
    # pick how text becomes vectors (see MEMORY_RUN.md):
    #   PowerShell:  $env:MEMORY_EMBED_PROVIDER="demo"      cmd:  set MEMORY_EMBED_PROVIDER=demo
    #   Mac/Linux:   export MEMORY_EMBED_PROVIDER=demo
    #   demo = no model needed (matches shared words only);  local = sentence-transformers;  gemini / openai = API key
    python run_memory_demo.py

It makes a throwaway case 'demo_case_xxxxxx', saves and searches memory in it, then deletes the case and
all of its memory again. Your existing cases are not touched.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import addDataById, connect_to_database, deleteDataById
from env_controller import getCaseDatabaseName
from structures import memory as m


def say(title: str) -> None:
    print(f"\n=== {title}")


def main() -> None:
    embedder = m.default_embedder()
    say(f"Embedding model: {embedder.name}")
    if embedder.name.startswith("demo:"):
        print("(demo mode: only shared WORDS match. Use local, gemini or openai for real meaning-based search.)")
    if not embedder(["probe"]):
        sys.exit("The embedding model is not available. Set MEMORY_EMBED_PROVIDER=demo to try without a model, "
                 "or local (pip install sentence-transformers) / gemini / openai with the right API key.")

    user, case_id = "demo_user", f"demo_case_{uuid.uuid4().hex[:6]}"
    addDataById(connect_to_database(), getCaseDatabaseName(),
                {"_id": case_id, "created_by": user, "title": "Memory demo case"})
    who = dict(user_id=user, case_id=case_id, embedder=embedder)
    try:
        say("1. Claim agent saves 3 claims (one embedding call for all three)")
        claims = ["Claim 1: A vacuum cleaner comprising a dust container and a cyclone separator.",
                  "Claim 2: The vacuum cleaner of claim 1, wherein the filter is washable.",
                  "Claim 3: A battery pack comprising a latch and a charging contact."]
        for result in m.save_memories(agent="claim_isolator_classifier", memory_type="claim", contents=claims,
                                      patent_id="US-DEMO-1", **who):
            print(" ", result)

        say("2. Product agent saves a product with its source")
        m.save_memory(agent="product_source_miner", memory_type="product", patent_id="US-DEMO-1", source_ref="https://example.com/vac-3000",
                      content="Acme Vac 3000: bagless vacuum with cyclone separation and a washable filter", **who)

        say("3. Permissions: the product agent tries to write a CLAIM")
        try:
            m.save_memory(agent="product_source_miner", memory_type="claim", content="Claim 9: something", **who)
        except PermissionError as e:
            print("  denied ->", e)

        say("4. Analysis agent searches the claims")
        for hit in m.retrieve_memory(user_id=user, agent="patent_infringement_analyzer", case_id=case_id,
                                     query="cyclone vacuum dust", memory_type="claim", k=3, embedder=embedder):
            print(f"  {hit['score']:.2f} [{hit['status']}] {hit['content'][:70]}")

        say("5. Analysis finding with evidence, then the verification agent checks it")
        finding = m.save_memory(agent="patent_infringement_analyzer", memory_type="analysis",
                                content="Acme Vac 3000 has a cyclone separator, matching claim 1",
                                evidence=[{"url": "https://example.com/vac-3000", "quote": "cyclone separation"}], **who)
        m.set_memory_status(user_id=user, agent="verification", memory_id=finding["memory_id"], status="verified",
                            verification={"check": "quote_found_in_source"})
        print("  verified-only search:", [h["content"][:50] for h in m.retrieve_memory(
            user_id=user, agent="report_builder", case_id=case_id, query="cyclone", verified_only=True, embedder=embedder)])

        say("6. Text an agent can put in its prompt")
        print(m.build_agent_context(user_id=user, agent="report_builder", case_id=case_id, query="washable filter cyclone",
                                    embedder=embedder))

        say("7. Correction: new version, old one kept as history")
        first = m.list_case_memories(user_id=user, agent="user", case_id=case_id, memory_type="claim")[0]
        newer = m.update_memory(user_id=user, agent="claim_isolator_classifier", memory_id=first["memory_id"],
                                content=claims[0] + " (corrected)", embedder=embedder)
        print("  new version:", newer)
        print("  summary:", m.memory_summary(user_id=user, agent="user", case_id=case_id))
    finally:
        say("Cleaning up the demo case")
        removed = m.delete_case_memories(user_id=user, agent="host", case_id=case_id)
        deleteDataById(connect_to_database(), getCaseDatabaseName(), case_id)
        print(f"  removed {removed} memories and the demo case")


if __name__ == "__main__":
    main()
