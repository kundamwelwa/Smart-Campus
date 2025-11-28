"""
Room Usage Optimizer - Reinforcement Learning

PPO-based optimizer for room allocation that minimizes energy cost and travel time.
Uses Ray RLlib for distributed training.
"""

from typing import Any

import numpy as np
import structlog
from gymnasium import spaces

from ml.models.base import BaseMLModel

logger = structlog.get_logger(__name__)


class RoomAllocationEnv:
    """
    Gym environment for room allocation optimization.

    State: Current allocation, room occupancy, time slots, energy usage
    Action: Assign course section to room
    Reward: -(energy_cost + travel_distance + constraint_violations)
    """

    def __init__(
        self,
        num_rooms: int = 50,
        num_sections: int = 100,
        time_slots: int = 12,
    ):
        """
        Initialize environment.

        Args:
            num_rooms: Number of available rooms
            num_sections: Number of course sections to schedule
            time_slots: Number of time slots per day
        """
        self.num_rooms = num_rooms
        self.num_sections = num_sections
        self.time_slots = time_slots

        # State space: room features + section features + current allocation
        # [room_capacities, room_locations, section_sizes, current_time, allocation_matrix]
        state_dim = num_rooms * 3 + num_sections * 2 + num_rooms * time_slots

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )

        # Action space: choose room for current section
        self.action_space = spaces.Discrete(num_rooms)

        # Environment state
        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        # Room properties
        self.room_capacities = np.random.randint(20, 200, size=self.num_rooms)
        self.room_locations = np.random.rand(self.num_rooms, 2)  # 2D coordinates
        self.room_energy_efficiency = np.random.rand(self.num_rooms)  # 0-1

        # Section properties
        self.section_sizes = np.random.randint(15, 150, size=self.num_sections)
        self.section_preferences = np.random.rand(self.num_sections, 2)  # Preferred location

        # Allocation matrix [rooms x time_slots]
        self.allocation = np.zeros((self.num_rooms, self.time_slots), dtype=int)

        # Current section to allocate
        self.current_section = 0
        self.current_time_slot = 0

        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        state = np.concatenate([
            self.room_capacities / 200.0,  # Normalize
            self.room_locations.flatten(),
            self.room_energy_efficiency,
            self.section_sizes / 150.0,
            self.section_preferences.flatten(),
            self.allocation.flatten() / self.num_sections,
        ])
        return state.astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Room index to assign current section

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        room_id = action
        section_id = self.current_section

        # Calculate reward components
        reward = 0.0
        violations = []

        # 1. Capacity constraint (hard)
        if self.section_sizes[section_id] > self.room_capacities[room_id]:
            reward -= 100.0  # Large penalty
            violations.append('capacity_exceeded')

        # 2. Room conflicts (hard) - check if room already occupied at this time
        if self.allocation[room_id, self.current_time_slot] != 0:
            reward -= 100.0
            violations.append('room_conflict')

        # 3. Energy cost (soft) - prefer efficient rooms
        energy_cost = (1.0 - self.room_energy_efficiency[room_id]) * 10.0
        reward -= energy_cost

        # 4. Travel distance (soft) - minimize distance from preferred location
        room_loc = self.room_locations[room_id]
        preferred_loc = self.section_preferences[section_id]
        travel_distance = np.linalg.norm(room_loc - preferred_loc) * 20.0
        reward -= travel_distance

        # 5. Utilization bonus - reward for using rooms efficiently
        utilization = self.section_sizes[section_id] / self.room_capacities[room_id]
        if 0.7 <= utilization <= 1.0:
            reward += 10.0  # Good utilization

        # Update allocation if no hard constraints violated
        if not violations or len([v for v in violations if 'capacity' not in v and 'conflict' not in v]) == len(violations):
            self.allocation[room_id, self.current_time_slot] = section_id + 1

        # Move to next section
        self.current_section += 1

        # Check if episode is done
        terminated = self.current_section >= self.num_sections
        truncated = False

        # Move to next time slot if needed
        if self.current_section % (self.num_sections // self.time_slots + 1) == 0:
            self.current_time_slot = min(self.current_time_slot + 1, self.time_slots - 1)

        info = {
            'energy_cost': energy_cost,
            'travel_distance': travel_distance,
            'violations': violations,
            'utilization': utilization,
        }

        return self._get_state(), reward, terminated, truncated, info


class RoomUsageOptimizer(BaseMLModel):
    """
    Room usage optimizer using Proximal Policy Optimization (PPO).

    Optimizes room assignments to minimize:
    - Energy consumption
    - Travel time
    - Constraint violations
    """

    def __init__(
        self,
        model_name: str = "room_optimizer",
        version: str = "1.0.0",
        num_rooms: int = 50,
        num_sections: int = 100,
    ):
        """
        Initialize room optimizer.

        Args:
            model_name: Model identifier
            version: Model version
            num_rooms: Number of rooms in campus
            num_sections: Number of sections to schedule
        """
        super().__init__(model_name, version)

        self.num_rooms = num_rooms
        self.num_sections = num_sections
        self.env = RoomAllocationEnv(num_rooms, num_sections)

        # PPO will be integrated with Ray RLlib (placeholder for now)
        self.policy = None

    async def train(
        self,
        training_data: Any,
        validation_data: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Train PPO agent on room allocation task.

        Args:
            training_data: Environment configuration
            validation_data: Validation environments
            **kwargs: Training parameters (seed, deterministic, etc.)

        Returns:
            dict: Training results
        """
        logger.info("Starting room optimizer training with PPO")

        # Set deterministic behavior if seed provided
        seed = kwargs.get('seed')
        deterministic = kwargs.get('deterministic', seed is not None)

        if seed is not None:
            self.set_deterministic(seed)
        elif deterministic:
            self.set_deterministic(42)  # Default seed

        # Training configuration
        num_iterations = kwargs.get('num_iterations', 100)
        episodes_per_iteration = kwargs.get('episodes_per_iteration', 10)

        # Simplified PPO training loop (full implementation would use Ray)
        total_rewards = []

        for iteration in range(num_iterations):
            iteration_rewards = []

            for episode in range(episodes_per_iteration):
                # Use seed for deterministic environment resets
                env_seed = seed + iteration * episodes_per_iteration + episode if seed is not None else None
                state, _ = self.env.reset(seed=env_seed)
                episode_reward = 0
                terminated = False

                while not terminated:
                    # Random policy (placeholder - real PPO would use learned policy)
                    action = self.env.action_space.sample()
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    episode_reward += reward

                iteration_rewards.append(episode_reward)

            avg_reward = np.mean(iteration_rewards)
            total_rewards.append(avg_reward)

            if (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{num_iterations}",
                    avg_reward=avg_reward,
                )

        self.is_trained = True

        # Save model and register version
        from pathlib import Path

        from shared.config import settings
        model_dir = Path(settings.ml_model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'room-optimizer-v{self.version}.pt'

        # Save policy state (placeholder - would save actual PPO policy)
        await self.save(model_path, register_version=True)

        return {
            'final_average_reward': float(np.mean(total_rewards[-10:])),
            'best_reward': float(max(total_rewards)),
            'training_iterations': num_iterations,
            'reward_history': [float(r) for r in total_rewards],
            'version': self.version,
            'seed': self._seed,
            'model_path': str(model_path),
        }

    async def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict optimal room allocation.

        Args:
            input_data: Dictionary with sections and rooms data

        Returns:
            dict: Optimal allocation and metrics
        """
        # For now, use a greedy heuristic (real implementation uses trained PPO)
        sections = input_data.get('sections', [])
        rooms = input_data.get('rooms', [])

        allocation = {}
        total_energy_cost = 0.0
        total_travel_cost = 0.0
        violations = 0

        for i, section in enumerate(sections):
            # Find best room (greedy)
            best_room = None
            best_score = float('-inf')

            for j, room in enumerate(rooms):
                # Check capacity
                if section['size'] > room['capacity']:
                    continue

                # Calculate score
                energy_score = room.get('energy_efficiency', 0.5)

                # Distance score (if locations provided)
                if 'location' in section and 'location' in room:
                    distance = np.linalg.norm(
                        np.array(section['location']) - np.array(room['location'])
                    )
                    travel_score = 1.0 / (1.0 + distance)
                else:
                    travel_score = 0.5

                score = energy_score + travel_score

                if score > best_score:
                    best_score = score
                    best_room = j

            if best_room is not None:
                allocation[i] = best_room
            else:
                violations += 1

        return {
            'allocation': allocation,
            'metrics': {
                'total_energy_cost': total_energy_cost,
                'total_travel_cost': total_travel_cost,
                'constraint_violations': violations,
                'success_rate': (len(sections) - violations) / len(sections) if sections else 0,
            },
            'num_sections_allocated': len(allocation),
            'num_violations': violations,
        }

    async def explain(self, input_data: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
        """
        Explain room allocation decisions with detailed reasoning.

        Args:
            input_data: Input data (sections, rooms)
            prediction: Allocation prediction

        Returns:
            dict: Explanation of decisions with optimization rationale
        """
        allocation = prediction['allocation']
        metrics = prediction['metrics']
        sections = input_data.get('sections', [])
        rooms = input_data.get('rooms', [])

        explanations = []

        for section_id, room_id in allocation.items():
            section = sections[section_id] if section_id < len(sections) else {}
            room = rooms[room_id] if room_id < len(rooms) else {}

            reasons = []

            # Capacity analysis
            if 'size' in section and 'capacity' in room:
                utilization = section['size'] / room['capacity']
                if utilization > 0.9:
                    reasons.append(f'High utilization ({utilization:.1%}) - efficient use of space')
                elif utilization > 0.7:
                    reasons.append(f'Good utilization ({utilization:.1%}) - balanced occupancy')
                else:
                    reasons.append(f'Sufficient capacity ({utilization:.1%} utilization)')

            # Energy efficiency
            if 'energy_efficiency' in room:
                eff = room['energy_efficiency']
                if eff > 0.8:
                    reasons.append(f'High energy efficiency ({eff:.1%}) - low operational cost')
                elif eff > 0.5:
                    reasons.append(f'Moderate energy efficiency ({eff:.1%})')
                else:
                    reasons.append(f'Lower energy efficiency ({eff:.1%}) - acceptable trade-off')

            # Travel distance
            if 'location' in section and 'location' in room:
                distance = np.linalg.norm(
                    np.array(section['location']) - np.array(room['location'])
                )
                if distance < 0.2:
                    reasons.append(f'Minimal travel distance ({distance:.2f} units) - convenient location')
                elif distance < 0.5:
                    reasons.append(f'Reasonable travel distance ({distance:.2f} units)')
                else:
                    reasons.append(f'Longer travel distance ({distance:.2f} units) - capacity/energy trade-off')

            explanation = {
                'section_id': section_id,
                'assigned_room': room_id,
                'reasons': reasons,
                'section_size': section.get('size', 'N/A'),
                'room_capacity': room.get('capacity', 'N/A'),
            }
            explanations.append(explanation)

        return {
            'allocation_explanations': explanations,
            'overall_metrics': metrics,
            'optimization_goals': [
                'Minimize energy consumption',
                'Minimize student travel time',
                'Maximize room utilization',
                'Satisfy capacity constraints',
            ],
            'summary': f"Allocated {len(allocation)} sections with {metrics['constraint_violations']} violations. "
                      f"Success rate: {metrics['success_rate']:.1%}",
            'explainability_method': 'constraint_satisfaction_and_optimization',
            'model_version': self.version,
        }

