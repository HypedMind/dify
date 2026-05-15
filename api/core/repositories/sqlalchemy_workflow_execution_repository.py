"""
SQLAlchemy implementation of the WorkflowExecutionRepository.
"""

import json
import logging
import queue
import threading
from typing import Optional, Union

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from core.workflow.entities.workflow_execution import (
    WorkflowExecution,
    WorkflowExecutionStatus,
    WorkflowType,
)
from core.workflow.repositories.workflow_execution_repository import WorkflowExecutionRepository
from core.workflow.workflow_type_encoder import WorkflowRuntimeTypeConverter
from libs.helper import extract_tenant_id
from models import (
    Account,
    CreatorUserRole,
    EndUser,
    WorkflowRun,
)
from models.enums import WorkflowRunTriggeredFrom

logger = logging.getLogger(__name__)


class SQLAlchemyWorkflowExecutionRepository(WorkflowExecutionRepository):
    """
    SQLAlchemy implementation of the WorkflowExecutionRepository interface.

    This implementation supports multi-tenancy by filtering operations based on tenant_id.
    Each method creates its own session, handles the transaction, and commits changes
    to the database. This prevents long-running connections in the workflow core.

    This implementation also includes an in-memory cache for workflow executions to improve
    performance by reducing database queries.
    """

    def __init__(
        self,
        session_factory: sessionmaker | Engine,
        user: Union[Account, EndUser],
        app_id: Optional[str],
        triggered_from: Optional[WorkflowRunTriggeredFrom],
    ):
        """
        Initialize the repository with a SQLAlchemy sessionmaker or engine and context information.

        Args:
            session_factory: SQLAlchemy sessionmaker or engine for creating sessions
            user: Account or EndUser object containing tenant_id, user ID, and role information
            app_id: App ID for filtering by application (can be None)
            triggered_from: Source of the execution trigger (DEBUGGING or APP_RUN)
        """
        # If an engine is provided, create a sessionmaker from it
        if isinstance(session_factory, Engine):
            self._session_factory = sessionmaker(bind=session_factory, expire_on_commit=False)
        elif isinstance(session_factory, sessionmaker):
            self._session_factory = session_factory
        else:
            raise ValueError(
                f"Invalid session_factory type {type(session_factory).__name__}; expected sessionmaker or Engine"
            )

        # Extract tenant_id from user
        tenant_id = extract_tenant_id(user)
        if not tenant_id:
            raise ValueError("User must have a tenant_id or current_tenant_id")
        self._tenant_id = tenant_id

        # Store app context
        self._app_id = app_id

        # Extract user context
        self._triggered_from = triggered_from
        self._creator_user_id = user.id

        # Determine user role based on user type
        self._creator_user_role = CreatorUserRole.ACCOUNT if isinstance(user, Account) else CreatorUserRole.END_USER

        # Initialize in-memory cache for workflow executions
        # Key: execution_id, Value: WorkflowRun (DB model)
        self._execution_cache: dict[str, WorkflowRun] = {}

        # Studio-local change: async workflow-execution persistence.
        # Same pattern as the node-execution repo — the first save (workflow
        # start) used to block the first node by ~100-200ms while round-tripping
        # PgBouncer. We enqueue and let a background worker commit; the cycle
        # manager calls flush() at workflow terminal handlers so the WorkflowRun
        # row is durable before the workflow-finished SSE event reads it back
        # (see workflow_response_converter.workflow_finish_to_stream_response).
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._shutdown = False

    def _to_domain_model(self, db_model: WorkflowRun) -> WorkflowExecution:
        """
        Convert a database model to a domain model.

        Args:
            db_model: The database model to convert

        Returns:
            The domain model
        """
        # Parse JSON fields
        inputs = db_model.inputs_dict
        outputs = db_model.outputs_dict
        graph = db_model.graph_dict

        # Convert status to domain enum
        status = WorkflowExecutionStatus(db_model.status)

        return WorkflowExecution(
            id_=db_model.id,
            workflow_id=db_model.workflow_id,
            workflow_type=WorkflowType(db_model.type),
            workflow_version=db_model.version,
            graph=graph,
            inputs=inputs,
            outputs=outputs,
            status=status,
            error_message=db_model.error or "",
            total_tokens=db_model.total_tokens,
            total_steps=db_model.total_steps,
            exceptions_count=db_model.exceptions_count,
            started_at=db_model.created_at,
            finished_at=db_model.finished_at,
        )

    def _to_db_model(self, domain_model: WorkflowExecution) -> WorkflowRun:
        """
        Convert a domain model to a database model.

        Args:
            domain_model: The domain model to convert

        Returns:
            The database model
        """
        # Use values from constructor if provided
        if not self._triggered_from:
            raise ValueError("triggered_from is required in repository constructor")
        if not self._creator_user_id:
            raise ValueError("created_by is required in repository constructor")
        if not self._creator_user_role:
            raise ValueError("created_by_role is required in repository constructor")

        db_model = WorkflowRun()
        db_model.id = domain_model.id_
        db_model.tenant_id = self._tenant_id
        if self._app_id is not None:
            db_model.app_id = self._app_id
        db_model.workflow_id = domain_model.workflow_id
        db_model.triggered_from = self._triggered_from

        # No sequence number generation needed anymore

        db_model.type = domain_model.workflow_type
        db_model.version = domain_model.workflow_version
        db_model.graph = json.dumps(domain_model.graph) if domain_model.graph else None
        db_model.inputs = json.dumps(domain_model.inputs) if domain_model.inputs else None
        db_model.outputs = (
            json.dumps(WorkflowRuntimeTypeConverter().to_json_encodable(domain_model.outputs))
            if domain_model.outputs
            else None
        )
        db_model.status = domain_model.status
        db_model.error = domain_model.error_message if domain_model.error_message else None
        db_model.total_tokens = domain_model.total_tokens
        db_model.total_steps = domain_model.total_steps
        db_model.exceptions_count = domain_model.exceptions_count
        db_model.created_by_role = self._creator_user_role
        db_model.created_by = self._creator_user_id
        db_model.created_at = domain_model.started_at
        db_model.finished_at = domain_model.finished_at

        # Calculate elapsed time if finished_at is available
        if domain_model.finished_at:
            db_model.elapsed_time = (domain_model.finished_at - domain_model.started_at).total_seconds()
        else:
            db_model.elapsed_time = 0

        return db_model

    def save(self, execution: WorkflowExecution) -> None:
        """
        Studio-local change: async workflow-execution persistence.

        Converts the domain entity to a DB model on the calling thread (cheap),
        updates the in-memory cache, and enqueues the DB write for a background
        worker. Call flush() before any code that reads the row back from DB
        (e.g. workflow-finished SSE event in workflow_response_converter).
        """
        db_model = self._to_db_model(execution)

        # Cache eagerly so subsequent reads against this repository see the
        # latest state even before the async write commits.
        self._execution_cache[db_model.id] = db_model

        self._ensure_writer_started()
        self._write_queue.put(db_model)

    def _ensure_writer_started(self) -> None:
        if self._writer_thread is not None:
            return
        with self._writer_lock:
            if self._writer_thread is not None:
                return
            t = threading.Thread(
                target=self._writer_loop,
                name=f"wf-exec-writer-{id(self)}",
                daemon=True,
            )
            t.start()
            self._writer_thread = t

    def _writer_loop(self) -> None:
        """
        Drain the write queue using a single long-lived session. Order is FIFO,
        so the start save commits before the finish save for the same row.
        """
        session = self._session_factory()
        try:
            while True:
                item = self._write_queue.get()
                try:
                    if item is None:
                        return
                    try:
                        session.merge(item)
                        session.commit()
                    except Exception:
                        logger.exception(
                            "Async workflow-execution write failed for id=%s",
                            getattr(item, "id", None),
                        )
                        try:
                            session.rollback()
                        except Exception:
                            logger.exception("Rollback failed in workflow-execution writer")
                finally:
                    self._write_queue.task_done()
        finally:
            try:
                session.close()
            except Exception:
                logger.exception("Failed to close workflow-execution writer session")

    def flush(self, timeout: Optional[float] = 30.0) -> None:
        """
        Block until all enqueued workflow-execution writes are durable.
        Must be called before any consumer reads the WorkflowRun row from DB
        (e.g. terminal SSE event emission).
        """
        if self._writer_thread is None:
            return
        if timeout is None:
            self._write_queue.join()
            return
        done = threading.Event()

        def _waiter() -> None:
            self._write_queue.join()
            done.set()

        threading.Thread(target=_waiter, name="wf-exec-flush-wait", daemon=True).start()
        if not done.wait(timeout):
            logger.warning(
                "Workflow-execution flush timed out after %.1fs; %d writes still pending",
                timeout,
                self._write_queue.unfinished_tasks,
            )

    def shutdown(self) -> None:
        """Stop the background writer. Safe to call repeatedly."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._writer_thread is None:
            return
        self._write_queue.put(None)
        self._writer_thread.join(timeout=30.0)
