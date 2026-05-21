from orchestration.tools import plan_comparison_search
from orchestration.tools import preload_retrievers

preload_retrievers()
res = plan_comparison_search.invoke({"query": "Compare the deductibles for Bronze and Gold plans.", "tier": "Bronze"})
print(res)
