"""HMARL maritime MVP package."""

from .agents import FleetCoordinatorAgent, PortAgent, VesselAgent
from .buffer import MultiAgentRolloutBuffer, RolloutBuffer
from .checkpointing import EarlyStopping, TrainingCheckpoint
from .config import (
    DISTANCE_NM,
    SEED,
    DecisionCadence,
    HMARLConfig,
    generate_distance_matrix,
    get_default_config,
    resolve_distance_matrix,
    validate_config,
    validate_distance_matrix,
)
from .curriculum import CurriculumScheduler, CurriculumStage
from .dynamics import (
    compute_fuel_and_emissions,
    dispatch_vessel,
    generate_weather,
    step_ports,
    step_vessels,
    update_weather_ar1,
    weather_fuel_multiplier,
    weather_speed_factor,
)
from .env import MaritimeEnv
from .experiment import (
    run_experiment,
    run_horizon_sweep,
    run_mappo_ablation,
    run_mappo_comparison,
    run_mappo_hyperparam_sweep,
    run_noise_sweep,
    run_policy_sweep,
    run_sharing_sweep,
    save_result_dict,
    summarize_policy_results,
)
from .experiment_config import ExperimentConfig, load_experiment_config, run_from_config
from .forecasts import MediumTermForecaster, OracleForecaster, ShortTermForecaster
from .gym_wrapper import MaritimeGymEnv
from .learned_forecaster import (
    ForecastDataset,
    LearnedForecaster,
    RNNForecastDataset,
    RNNForecaster,
    TrainResult,
    build_forecast_dataset,
    build_rnn_dataset,
    collect_expanded_queue_traces,
    collect_queue_traces,
    train_forecaster,
    train_rnn_forecaster,
)
from .logger import TrainingLogger
from .mappo import MAPPOConfig, MAPPOTrainer, PPOUpdateResult, train_multi_seed
from .message_bus import MessageBus
from .metrics import (
    compute_coordination_metrics,
    compute_coordinator_metrics,
    compute_economic_metrics,
    compute_economic_step_deltas,
    compute_port_metrics,
    compute_vessel_metrics,
    forecast_mae,
    forecast_rmse,
)
from .networks import ActorCritic, build_actor_critics, build_per_agent_actor_critics
from .plotting import (
    plot_ablation_bar,
    plot_horizon_sweep,
    plot_mappo_comparison,
    plot_multi_seed_curves,
    plot_noise_sweep,
    plot_policy_comparison,
    plot_sharing_sweep,
    plot_sweep_heatmap,
    plot_timing_breakdown,
    plot_training_curves,
    plot_training_dashboard,
)
from .policies import FleetCoordinatorPolicy, PortPolicy, VesselPolicy
from .report import generate_training_report
from .rewards import (
    compute_coordinator_reward_step,
    compute_port_reward,
    compute_vessel_reward_step,
    weather_coordinator_shaping,
    weather_vessel_shaping,
)
from .state import PortState, VesselState, initialize_ports, initialize_vessels, make_rng
from .stats import bootstrap_ci, compare_methods, multi_method_comparison, welch_t_test

__all__ = [
    # config
    "DISTANCE_NM",
    "SEED",
    "DecisionCadence",
    "HMARLConfig",
    "generate_distance_matrix",
    "get_default_config",
    "resolve_distance_matrix",
    "validate_config",
    "validate_distance_matrix",
    # state
    "PortState",
    "VesselState",
    "initialize_ports",
    "initialize_vessels",
    "make_rng",
    # environment
    "MaritimeEnv",
    "MaritimeGymEnv",
    # agents
    "FleetCoordinatorAgent",
    "PortAgent",
    "VesselAgent",
    # policies
    "FleetCoordinatorPolicy",
    "PortPolicy",
    "VesselPolicy",
    # MAPPO
    "MAPPOConfig",
    "MAPPOTrainer",
    "PPOUpdateResult",
    "train_multi_seed",
    # networks
    "ActorCritic",
    "build_actor_critics",
    "build_per_agent_actor_critics",
    # buffers
    "MultiAgentRolloutBuffer",
    "RolloutBuffer",
    # curriculum
    "CurriculumScheduler",
    "CurriculumStage",
    # experiment
    "ExperimentConfig",
    "load_experiment_config",
    "run_from_config",
    "run_experiment",
    "run_horizon_sweep",
    "run_mappo_comparison",
    "run_mappo_hyperparam_sweep",
    "run_mappo_ablation",
    "run_noise_sweep",
    "run_policy_sweep",
    "run_sharing_sweep",
    "save_result_dict",
    "summarize_policy_results",
    # stats
    "bootstrap_ci",
    "compare_methods",
    "multi_method_comparison",
    "welch_t_test",
    # metrics
    "compute_coordination_metrics",
    "compute_coordinator_metrics",
    "compute_economic_metrics",
    "compute_economic_step_deltas",
    "compute_port_metrics",
    "compute_vessel_metrics",
    "forecast_mae",
    "forecast_rmse",
    # rewards
    "compute_coordinator_reward_step",
    "compute_port_reward",
    "compute_vessel_reward_step",
    "weather_coordinator_shaping",
    "weather_vessel_shaping",
    # forecasts
    "MediumTermForecaster",
    "OracleForecaster",
    "ShortTermForecaster",
    # learned forecaster
    "ForecastDataset",
    "LearnedForecaster",
    "TrainResult",
    "build_forecast_dataset",
    "collect_queue_traces",
    "train_forecaster",
    # dynamics
    "compute_fuel_and_emissions",
    "dispatch_vessel",
    "generate_weather",
    "step_ports",
    "step_vessels",
    "update_weather_ar1",
    "weather_fuel_multiplier",
    "weather_speed_factor",
    # plotting
    "plot_ablation_bar",
    "plot_horizon_sweep",
    "plot_mappo_comparison",
    "plot_multi_seed_curves",
    "plot_noise_sweep",
    "plot_policy_comparison",
    "plot_sharing_sweep",
    "plot_sweep_heatmap",
    "plot_timing_breakdown",
    "plot_training_curves",
    "plot_training_dashboard",
    # infrastructure
    "MessageBus",
    "TrainingLogger",
    "EarlyStopping",
    "TrainingCheckpoint",
    "generate_training_report",
]
