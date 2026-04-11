"""Public exception hierarchy for adk-lazy-mcp."""


class LazyMCPError(Exception):
    """Base exception for wrapper-level failures."""


class RegistryClosedError(LazyMCPError):
    """Raised when a call is attempted after registry shutdown."""


class PolicyDeniedError(LazyMCPError):
    """Raised when policy denies an action."""


class ValidationError(LazyMCPError):
    """Raised when client-side JSON schema validation fails."""


class ToolNotFoundError(LazyMCPError):
    """Raised when a server/tool is missing from the catalog."""
