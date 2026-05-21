import asyncio
from orchestration.orchestrator import Orchestrator

def main():
    orch = Orchestrator()
    res = orch.ask_detailed("Compare the deductibles for Bronze and Gold plans.")
    print("ANSWER:")
    print(res["answer"])
    print("\nCONTEXT:")
    print(res["retrieved_context"])
    print("\nSTEPS LOG:")
    print("\n".join(res["steps_log"]))

main()
