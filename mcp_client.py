"""Semrush MCP server client for Vertex AI agent.

This client connects to the hosted Semrush MCP server at https://mcp.semrush.com/v1/mcp
and handles OAuth authentication and action execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from cost_estimator import CostEstimate, estimate_cost

MCP_ENDPOINT = "https://mcp.semrush.com/v1/mcp"
BALANCE_ENDPOINT = "https://www.semrush.com/users/countapiunits.html"

# Map our action names to Semrush MCP toolkit and report names
ACTION_TO_TOOLKIT_REPORT = {
    "get_domain_overview": ("overview", "domain_rank"),
    "get_keyword_overview": ("keyword", "phrase_this"),
    "get_keyword_fullsearch": ("keyword", "phrase_fullsearch"),
    "get_keyword_related": ("keyword", "phrase_related"),
    "get_keyword_organic": ("keyword", "phrase_organic"),
    "get_keyword_questions": ("keyword", "phrase_questions"),
    "get_keyword_difficulty": ("keyword", "phrase_kdi"),
    "get_organic_keywords": ("domains", "domain_organic"),
    "get_competitors": ("domains", "domain_organic_organic"),
    "get_backlinks": ("backlinks", "backlinks"),
    "get_traffic_summary": ("trends", "traffic_summary"),
    "get_daily_traffic": ("trends", "daily_traffic"),
    "get_audience_insights": ("trends", "audience_insights"),
    "list_projects": ("projects", "list_projects"),
    "get_project": ("projects", "get_project"),
    "get_position_tracking": ("tracking", "tracking_position_organic"),
    "get_site_audit": ("siteaudit", "meta_issues"),
    "get_locations": ("projects", "locations"),
    "get_map_rank_tracker": ("projects", "map_rank_tracker"),
    # check_api_balance is not a Semrush MCP tool - handled separately
}

# Map parameter names from our action schema to Semrush MCP parameter names
# Some actions use different parameter names than what the MCP server expects
PARAMETER_MAPPING = {
    "get_keyword_overview": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    "get_keyword_fullsearch": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    "get_keyword_related": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    "get_keyword_organic": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    "get_keyword_questions": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    "get_keyword_difficulty": {"keyword": "phrase"},  # MCP expects "phrase" not "keyword"
    # Add other mappings as needed
}


@dataclass
class MCPActionResult:
    """Result from an MCP action execution."""

    action: str
    params: Dict[str, Any]
    data: Any
    balance_before: Optional[int] = None
    balance_after: Optional[int] = None
    estimated_cost: Optional[CostEstimate] = None
    error: Optional[str] = None

    @property
    def units_used(self) -> Optional[int]:
        """Calculate API units used."""
        if self.balance_before is None or self.balance_after is None:
            return None
        return self.balance_before - self.balance_after

    def to_tool_response(self) -> Dict[str, Any]:
        """Convert to tool response format for Vertex AI."""
        payload: Dict[str, Any] = {
            "action": self.action,
            "params": self.params,
            "success": self.error is None,
        }

        if self.error:
            payload["error"] = self.error
            payload["message"] = f"Action '{self.action}' failed: {self.error}"
        else:
            payload["data"] = self.data
            payload["message"] = f"Action '{self.action}' completed successfully"

        if self.balance_before is not None:
            payload["balance_before"] = self.balance_before
        if self.balance_after is not None:
            payload["balance_after"] = self.balance_after
        if self.units_used is not None:
            payload["units_used"] = self.units_used
        if self.estimated_cost is not None:
            payload["estimated_cost"] = {
                "price_per_line": self.estimated_cost.price_per_line,
                "lines": self.estimated_cost.lines,
                "total": self.estimated_cost.total,
            }

        return payload


class SemrushMCPClient:
    """Client for Semrush MCP server with API key authentication."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        session: Optional[requests.Session] = None,
        track_balance: bool = True,
    ) -> None:
        """Initialize MCP client.

        Args:
            api_key: Semrush API key for MCP server authentication (required)
            session: Optional requests session for connection pooling
            track_balance: Whether to track API unit balance
        """
        self.api_key = api_key or os.getenv("SEMRUSH_API_KEY")
        self.session = session or requests.Session()
        self.track_balance = track_balance

        if not self.api_key:
            raise RuntimeError(
                "SEMRUSH_API_KEY is required for MCP server authentication. "
                "Set it as an environment variable or pass it to the constructor. "
                "Get your API key from: https://www.semrush.com/kb/92-api-key"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with API key authentication."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Apikey {self.api_key}",
        }

    def check_balance(self) -> Optional[int]:
        """Check API unit balance using direct API call."""
        if not self.api_key or not self.track_balance:
            return None

        try:
            response = self.session.get(
                BALANCE_ENDPOINT,
                params={"key": self.api_key},
                timeout=30,
            )
            response.raise_for_status()
            return int(response.text.strip())
        except Exception:
            return None

    def call(self, action_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method that wraps execute_action and returns tool response format."""
        result = self.execute_action(action_name, params)
        return result.to_tool_response()

    def execute_action(
        self, action_name: str, params: Dict[str, Any]
    ) -> MCPActionResult:
        """Execute an action via the Semrush MCP server.

        Args:
            action_name: Name of the action (e.g., "get_domain_overview")
            params: Action parameters

        Returns:
            MCPActionResult with data, balance tracking, and cost estimate
        """
        # Estimate cost before execution
        cost = estimate_cost(
            action_name, display_limit=params.get("display_limit")
        )

        # Check balance before
        balance_before = self.check_balance()

        # Handle special case: check_api_balance is not an MCP tool
        if action_name == "check_api_balance":
            balance = self.check_balance()
            return MCPActionResult(
                action=action_name,
                params=params,
                data={"balance": balance} if balance is not None else None,
                balance_before=balance,
                balance_after=balance,
                estimated_cost=None,
                error=None if balance is not None else "Could not check balance",
            )

        # Map action to toolkit and report
        toolkit_report = ACTION_TO_TOOLKIT_REPORT.get(action_name)
        if not toolkit_report:
            return MCPActionResult(
                action=action_name,
                params=params,
                data=None,
                balance_before=balance_before,
                balance_after=self.check_balance(),
                estimated_cost=cost,
                error=f"Action '{action_name}' not mapped to Semrush MCP toolkit/report",
            )

        toolkit, report = toolkit_report

        # Map parameters if needed (e.g., keyword -> phrase for phrase_this report)
        mapped_params = params.copy()
        if action_name in PARAMETER_MAPPING:
            for old_name, new_name in PARAMETER_MAPPING[action_name].items():
                if old_name in mapped_params:
                    mapped_params[new_name] = mapped_params.pop(old_name)

        # Build MCP request payload
        # Semrush MCP uses tools/call with semrush_report tool
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "semrush_report",
                "arguments": {
                    "toolkit": toolkit,
                    "report": report,
                    **mapped_params,  # Use mapped params instead of original params
                },
            },
            "id": 1,
        }

        try:
            # Execute via MCP server
            response = self.session.post(
                MCP_ENDPOINT,
                headers=self._get_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result_data = response.json()
            
            import sys
            print(f"[DEBUG] MCP Response: {result_data}", file=sys.stderr)

            # Parse MCP response
            # Semrush MCP returns: {"jsonrpc": "2.0", "id": 1, "result": {"content": [...], "isError": bool}}
            if "error" in result_data:
                error_msg = result_data["error"].get("message", "Unknown error")
                return MCPActionResult(
                    action=action_name,
                    params=params,
                    data=None,
                    balance_before=balance_before,
                    balance_after=self.check_balance(),
                    estimated_cost=cost,
                    error=error_msg,
                )

            # Extract result from MCP response
            result_obj = result_data.get("result", {})
            
            # Check if result indicates an error
            if result_obj.get("isError", False):
                # Extract error message from content
                content = result_obj.get("content", [])
                error_text = ""
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            error_text = item.get("text", "")
                            break
                
                return MCPActionResult(
                    action=action_name,
                    params=params,
                    data=None,
                    balance_before=balance_before,
                    balance_after=self.check_balance(),
                    estimated_cost=cost,
                    error=error_text or "MCP server returned an error",
                )

            # Extract data from content
            content = result_obj.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                # For now, extract text content - may need to handle other formats
                data = content
            else:
                data = result_obj

            # Check balance after
            balance_after = self.check_balance()

            return MCPActionResult(
                action=action_name,
                params=params,
                data=data,
                balance_before=balance_before,
                balance_after=balance_after,
                estimated_cost=cost,
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_msg = "API key invalid or expired. Please check your SEMRUSH_API_KEY."
            else:
                try:
                    error_body = e.response.json()
                    error_msg = f"HTTP {e.response.status_code}: {error_body}"
                except:
                    error_msg = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            
            import sys
            print(f"[DEBUG] MCP HTTP Error: {error_msg}", file=sys.stderr)

            return MCPActionResult(
                action=action_name,
                params=params,
                data=None,
                balance_before=balance_before,
                balance_after=self.check_balance(),
                estimated_cost=cost,
                error=error_msg,
            )

        except Exception as e:
            import sys
            print(f"[DEBUG] MCP Exception: {str(e)}", file=sys.stderr)
            return MCPActionResult(
                action=action_name,
                params=params,
                data=None,
                balance_before=balance_before,
                balance_after=self.check_balance(),
                estimated_cost=cost,
                error=f"Request failed: {str(e)}",
            )

