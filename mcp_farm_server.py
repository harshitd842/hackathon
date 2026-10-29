"""
mcp_farm_server.py — expose the local farm tools to a local SLM via MCP.

This is the "agentic" layer. A small local language model (e.g. Qwen2.5-3B via
Ollama or the Kompact AI runtime) can call these tools to turn a diagnosis into
an economic decision — completely offline.

Setup (optional, for the agentic demo):
    pip install mcp
    python seed_inventory.py          # create the local DB first
    python mcp_farm_server.py         # runs the MCP server on localhost

The two tools mirror the functions in farm_tools.py.
"""
from mcp.server.fastmcp import FastMCP

import farm_tools

mcp = FastMCP("Local Farm Inventory Agent")


@mcp.tool()
def check_pesticide_stock(chemical_name: str) -> str:
    """Check whether a pesticide/fungicide is already in the farmer's local shed, and its price."""
    return farm_tools.check_pesticide_stock(chemical_name)


@mcp.tool()
def calculate_treatment_roi(acreage: float, severity_pct: float) -> str:
    """Estimate the crop loss if untreated versus the cost of spraying, and whether spraying is worth it."""
    r = farm_tools.calculate_roi(acreage, severity_pct)
    return (
        f"Estimated loss if untreated: ₹{r['estimated_loss_inr']:.0f}; "
        f"treatment cost: ₹{r['treatment_cost_inr']:.0f}; "
        f"net benefit: ₹{r['net_benefit_inr']:.0f}; "
        f"spray_recommended={r['spray_recommended']}"
    )


if __name__ == "__main__":
    mcp.run()
