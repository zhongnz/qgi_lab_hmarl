"""HMARL maritime MVP package."""

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .buffer import MultiAgentRolloutBuffer, RolloutBuffer
from .checkpointing import EarlyStopping, TrainingCheckpoint
from .config import DISTANCE_NM, SEED, DecisionCadence, HMARLConfig, get_default_config
from .curriculum import CurriculumScheduler, CurriculumStage
from .dynamics import compute_fuel_and_emissions, generate_weather, update_weather_ar1
from .env import MaritimeEnv
from .experiment import (
    run_experiment,
    run_mappo_ablation,
    run_mappo_comparison,
    run_mappo_hyperparam_sweep,
    run_policy_sweep,
)
from .experiment_config import ExperimentConfig, load_experiment_config, run_from_config
from .forecasts import MediumTermForecaster, ShortTermForecaster
from .gym_wrapper import MaritimeGymEnv
from .logger import TrainingLogger
from .mappo import MAPPOConfig, MAPPOTrainer, PPOUpdateResult, train_multi_seed
from .metrics import (
    compute_coordinator_metrics,
    compute_economic_metrics,
    compute_port_metrics,
    compute_vessel_metrics,
)
from .networks import ActorCritic, build_actor_critics, build_per_agent_actor_critics
from .plotting import (
    plot_mappo_comparison,
    plot_multi_seed_curves,
    plot_policy_comparison,
    plot_sweep_heatmap,
    plot_training_curves,
    plot_training_dashboard,
)
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .rewards import (
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_vessel_reward_step,
)
from .state import PortState, VesselState, initialize_ports, initialize_vessels, make_rng
from .stats import bootstrap_ci, compare_methods, multi_method_comparison, welch_t_test

__all__ = [
    "DISTANCE_NM",
    "SEED",
    "DecisionCadence",
    "HMARLConfig",
    "get_default_config",
    "PortState",
    "VesselState",
    "initialize_ports",
    "initialize_vessels",
    "make_rng",
    "MaritimeEnv",
    "MaritimeGymEnv",
    "FleetCoordinatorAgent",
    "PortAgent",
    "VesselAgent",
    "FleetCoordinatorPolicy",
    "PortPolicy",
    "VesselPolicy",
    "MAPPOConfig",
    "MAPPOTrainer",
    "PPOUpdateResult",
    "train_multi_seed",
    "ActorCritic",
    "build_actor_critics",
    "build_per_agent_actor_critics",
    "MultiAgentRolloutBuffer",
    "RolloutBuffer",
    "CurriculumScheduler",
    "CurriculumStage",
    "ExperimentConfig",
    "load_experiment_config",
    "run_from_config",
    "run_experiment",
    "run_mappo_comparison",
    "run_mappo_hyperparam_sweep",
    "run_mappo_ablation",
    "run_policy_sweep",
    "bootstrap_ci",
    "compare_methods",
    "multi_method_comparison",
    "welch_t_test",
    "compute_economic_metrics",
    "compute_port_metrics",
    "compute_vessel_metrics",
    "compute_coordinator_metrics",
    "compute_coordinator_reward_step",
    "compute_port_reward",
    "compute_vessel_reward_step",
    "MediumTermForecaster",
    "ShortTermForecaster",
    "plot_mappo_comparison",
    "plot_multi_seed_curves",
    "plot_policy_comparison",
    "plot_sweep_heatmap",
    "plot_training_curves",
    "plot_training_dashboard",
    "TrainingLogger",
    "EarlyStopping",
    "TrainingCheckpoint",
    "compute_fuel_and_emissions",
    "generate_weather",
    "update_weather_ar1",
]
