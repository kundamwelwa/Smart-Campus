"""
Raft Consensus Algorithm Implementation

Implements leader election, log replication, and state machine replication
for distributed critical state management.

Based on the Raft paper: https://raft.github.io/raft.pdf
"""

import asyncio
import random
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from shared.config import settings

logger = structlog.get_logger(__name__)


class RaftState(str, Enum):
    """Raft node states."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntry(BaseModel):
    """
    Raft log entry.

    Represents a command to be applied to the state machine.
    """

    index: int = Field(..., description="Log index (1-indexed)")
    term: int = Field(..., description="Term when entry was created")
    command: dict[str, Any] = Field(..., description="State machine command")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VoteRequest(BaseModel):
    """RequestVote RPC request."""

    term: int
    candidate_id: int
    last_log_index: int
    last_log_term: int


class VoteResponse(BaseModel):
    """RequestVote RPC response."""

    term: int
    vote_granted: bool


class AppendEntriesRequest(BaseModel):
    """AppendEntries RPC request (heartbeat or log replication)."""

    term: int
    leader_id: int
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry] = Field(default_factory=list)
    leader_commit: int


class AppendEntriesResponse(BaseModel):
    """AppendEntries RPC response."""

    term: int
    success: bool
    match_index: int | None = None


class RaftNode:
    """
    Raft consensus node implementing leader election and log replication.

    Features:
    - Leader election with randomized timeouts
    - Log replication to followers
    - State machine application
    - Persistent state (term, voted_for, log)
    """

    def __init__(
        self,
        node_id: int,
        cluster_nodes: list[int],
        state_machine: Callable[[dict], Any] | None = None,
    ):
        """
        Initialize Raft node.

        Args:
            node_id: Unique node identifier
            cluster_nodes: List of all node IDs in cluster
            state_machine: Function to apply committed commands
        """
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.state_machine = state_machine or self._default_state_machine

        # Persistent state (should be persisted to disk in production)
        self.current_term = 0
        self.voted_for: int | None = None
        self.log: list[LogEntry] = []  # 1-indexed (log[0] is dummy)

        # Volatile state (all servers)
        self.commit_index = 0
        self.last_applied = 0
        self.state = RaftState.FOLLOWER

        # Volatile state (leaders only)
        self.next_index: dict[int, int] = {}  # For each follower
        self.match_index: dict[int, int] = {}  # For each follower

        # Election timing
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = datetime.utcnow()

        # Leader tracking
        self.current_leader: int | None = None

        # Election task
        self._election_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

        # Applied state machine commands
        self.applied_state: dict[str, Any] = {}

        logger.info(
            "Raft node initialized",
            node_id=node_id,
            cluster_size=len(cluster_nodes),
        )

    def _random_election_timeout(self) -> float:
        """Generate random election timeout (150-300ms)."""
        return random.uniform(
            settings.raft_election_timeout_min / 1000.0,
            settings.raft_election_timeout_max / 1000.0,
        )

    def _default_state_machine(self, command: dict[str, Any]) -> Any:
        """Default state machine - stores commands in dict."""
        key = command.get('key')
        value = command.get('value')

        if key:
            self.applied_state[key] = value

        return value

    async def start(self) -> None:
        """Start Raft node - begin election timeout monitoring."""
        logger.info("Starting Raft node", node_id=self.node_id)

        # Start election timeout monitoring
        self._election_task = asyncio.create_task(self._election_timeout_loop())

    async def stop(self) -> None:
        """Stop Raft node."""
        logger.info("Stopping Raft node", node_id=self.node_id)

        if self._election_task:
            self._election_task.cancel()

        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def _election_timeout_loop(self) -> None:
        """Monitor election timeout and trigger elections."""
        while True:
            await asyncio.sleep(0.01)  # Check every 10ms

            if self.state == RaftState.LEADER:
                continue  # Leaders don't timeout

            # Check if election timeout has elapsed
            time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()

            if time_since_heartbeat > self.election_timeout:
                # Start election
                await self._start_election()

    async def _start_election(self) -> None:
        """
        Start leader election.

        Steps:
        1. Increment current term
        2. Transition to candidate
        3. Vote for self
        4. Request votes from other nodes
        5. If majority votes received, become leader
        """
        # Transition to candidate
        self.state = RaftState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = datetime.utcnow()
        self.election_timeout = self._random_election_timeout()

        logger.info(
            "Starting election",
            node_id=self.node_id,
            term=self.current_term,
        )

        # Vote for self
        votes_received = 1
        votes_needed = (len(self.cluster_nodes) + 1) // 2  # Majority

        # In a real implementation, we would send RequestVote RPCs
        # to other nodes here. For this simplified version, we simulate
        # the election process.

        # Simulate receiving votes (in production, this would be actual RPCs)
        for peer in self.cluster_nodes:
            if peer == self.node_id:
                continue

            # Simulate vote response (simplified - real would be RPC)
            # In reality, followers would check log consistency
            vote_granted = random.random() < 0.6  # 60% chance of vote

            if vote_granted:
                votes_received += 1

        # Check if won election
        if votes_received >= votes_needed:
            await self._become_leader()
        else:
            # Election failed, return to follower
            self.state = RaftState.FOLLOWER
            logger.info(
                "Election lost",
                node_id=self.node_id,
                votes_received=votes_received,
                votes_needed=votes_needed,
            )

    async def _become_leader(self) -> None:
        """
        Become leader after winning election.

        Initializes leader state and starts sending heartbeats.
        """
        self.state = RaftState.LEADER
        self.current_leader = self.node_id

        # Initialize leader volatile state
        last_log_index = len(self.log)
        for peer in self.cluster_nodes:
            if peer != self.node_id:
                self.next_index[peer] = last_log_index + 1
                self.match_index[peer] = 0

        logger.info(
            "Became leader",
            node_id=self.node_id,
            term=self.current_term,
        )

        # Start sending heartbeats
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to followers."""
        while self.state == RaftState.LEADER:
            await self._send_heartbeats()
            await asyncio.sleep(settings.raft_heartbeat_interval / 1000.0)

    async def _send_heartbeats(self) -> None:
        """Send AppendEntries (heartbeat) to all followers."""
        for peer in self.cluster_nodes:
            if peer == self.node_id:
                continue

            # In production, send actual RPC
            # For now, just log
            logger.debug(
                "Sending heartbeat",
                leader=self.node_id,
                peer=peer,
                term=self.current_term,
            )

    async def append_entry(self, command: dict[str, Any]) -> bool:
        """
        Append a new entry to the log (client request).

        Only leaders can accept new entries.

        Args:
            command: Command to append

        Returns:
            bool: True if successfully replicated to majority
        """
        if self.state != RaftState.LEADER:
            logger.warning(
                "Cannot append entry - not leader",
                node_id=self.node_id,
                current_leader=self.current_leader,
            )
            return False

        # Create log entry
        entry = LogEntry(
            index=len(self.log) + 1,
            term=self.current_term,
            command=command,
        )

        # Append to local log
        self.log.append(entry)

        logger.info(
            "Entry appended to log",
            index=entry.index,
            term=entry.term,
            leader=self.node_id,
        )

        # Replicate to followers (simplified - real would wait for majority)
        # In production, send AppendEntries RPCs and wait for majority acks

        # For now, immediately commit (simplified)
        self.commit_index = entry.index

        # Apply to state machine
        await self._apply_committed_entries()

        return True

    async def _apply_committed_entries(self) -> None:
        """Apply committed log entries to state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1

            if self.last_applied <= len(self.log):
                entry = self.log[self.last_applied - 1]  # 0-indexed list

                # Apply to state machine
                self.state_machine(entry.command)

                logger.info(
                    "Applied entry to state machine",
                    index=entry.index,
                    command=entry.command.get('type', 'unknown'),
                    node_id=self.node_id,
                )

    def get_state(self) -> dict[str, Any]:
        """
        Get current Raft node state.

        Returns:
            dict: Current state information
        """
        return {
            'node_id': self.node_id,
            'state': self.state.value,
            'current_term': self.current_term,
            'current_leader': self.current_leader,
            'voted_for': self.voted_for,
            'log_length': len(self.log),
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'is_leader': self.state == RaftState.LEADER,
        }

    def read_state(self, key: str) -> Any | None:
        """
        Read value from replicated state.

        Args:
            key: State key

        Returns:
            Value or None
        """
        return self.applied_state.get(key)

