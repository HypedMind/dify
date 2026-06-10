from typing import Optional, Protocol

from core.workflow.entities.workflow_execution import WorkflowExecution


class WorkflowExecutionRepository(Protocol):
    """
    Repository interface for WorkflowExecution.

    This interface defines the contract for accessing and manipulating
    WorkflowExecution data, regardless of the underlying storage mechanism.

    Note: Domain-specific concepts like multi-tenancy (tenant_id), application context (app_id),
    and other implementation details should be handled at the implementation level, not in
    the core interface. This keeps the core domain model clean and independent of specific
    application domains or deployment scenarios.
    """

    def save(self, execution: WorkflowExecution) -> None:
        """
        Save or update a WorkflowExecution instance.

        Implementations may persist asynchronously; callers that need the write
        to be durable before proceeding (e.g. before emitting a SSE event that
        triggers a DB read of this row) should call ``flush()`` afterwards.

        Args:
            execution: The WorkflowExecution instance to save or update
        """
        ...

    def flush(self, timeout: Optional[float] = 30.0) -> None:
        """
        Block until all enqueued saves are durable.
        No-op default for synchronous repositories.
        """
        return None
