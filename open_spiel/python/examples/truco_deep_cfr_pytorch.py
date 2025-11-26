#!/usr/bin/env python3
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deep CFR training for Truco using PyTorch.

Deep CFR uses neural networks to approximate regret and strategy functions,
making it suitable for large games like Truco that cannot be solved with
tabular methods.

Example usage:
  # Quick test (5 minutes)
  python truco_deep_cfr_pytorch.py --num_iterations=10 --num_traversals=100
  
  # Standard training (1-2 hours)
  python truco_deep_cfr_pytorch.py --num_iterations=100 --num_traversals=500
  
  # Long training (overnight)
  python truco_deep_cfr_pytorch.py --num_iterations=1000 --num_traversals=1000
"""

from absl import app
from absl import flags
from absl import logging
import os
import pickle
import time
import sys

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel
from open_spiel.python.pytorch import deep_cfr

# Increase recursion depth for deep games like Truco
sys.setrecursionlimit(5000)

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, 
                     "Number of traversals/games per iteration")
flags.DEFINE_list("policy_network_layers", [128, 64],
                  "Layer sizes for policy network")
flags.DEFINE_list("advantage_network_layers", [128, 64],
                  "Layer sizes for advantage network")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
flags.DEFINE_integer("batch_size_advantage", 128,
                     "Batch size for advantage network training")
flags.DEFINE_integer("batch_size_strategy", 128,
                     "Batch size for strategy network training")
flags.DEFINE_integer("memory_capacity", int(1e7),
                     "Replay buffer capacity")
flags.DEFINE_integer("print_freq", 10, "Print progress every N iterations")
flags.DEFINE_integer("checkpoint_freq", 50, 
                     "Save checkpoint every N iterations (0=disable)")
flags.DEFINE_string("output_dir", "/tmp/truco_deep_cfr",
                    "Directory to save checkpoints and final policy")
flags.DEFINE_bool("verbose", True, "Print detailed progress")
flags.DEFINE_bool("compute_nash_conv", False, "Compute NashConv (exploitability) after training")


class TrucoPolicy:
  """A policy that uses the Deep CFR solver."""
  def __init__(self, solver):
    self.solver = solver

  def __call__(self, state):
    return self.solver.action_probabilities(state)

def main(unused_argv):
  """Train Deep CFR agent on Truco game."""
  start_time = time.time()
  
  logging.info("Loading Truco game...")
  game = pyspiel.load_game("truco")
  
  if FLAGS.verbose:
    logging.info("Game info:")
    logging.info("  Players: %d", game.num_players())
    logging.info("  Max game length: %d", game.max_game_length())
    logging.info("  Information state tensor shape: %s",
                 game.information_state_tensor_shape())
    logging.info("  Observation tensor shape: %s",
                 game.observation_tensor_shape())
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Convert layer sizes from strings to ints
  policy_layers = tuple(int(x) for x in FLAGS.policy_network_layers)
  advantage_layers = tuple(int(x) for x in FLAGS.advantage_network_layers)
  
  logging.info("\nInitializing Deep CFR solver...")
  logging.info("  Policy network: %s", policy_layers)
  logging.info("  Advantage network: %s", advantage_layers)
  logging.info("  Learning rate: %f", FLAGS.learning_rate)
  logging.info("  Batch size (advantage): %d", FLAGS.batch_size_advantage)
  logging.info("  Batch size (strategy): %d", FLAGS.batch_size_strategy)
  logging.info("  Memory capacity: %d", FLAGS.memory_capacity)
  
  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=policy_layers,
      advantage_network_layers=advantage_layers,
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=FLAGS.learning_rate,
      batch_size_advantage=FLAGS.batch_size_advantage,
      batch_size_strategy=FLAGS.batch_size_strategy,
      memory_capacity=FLAGS.memory_capacity)
  
  logging.info("\nStarting training...")
  logging.info("  Iterations: %d", FLAGS.num_iterations)
  logging.info("  Traversals per iteration: %d", FLAGS.num_traversals)
  logging.info("  Estimated time: %d-%d minutes",
               FLAGS.num_iterations // 10,
               FLAGS.num_iterations // 5)
  
  # Train manually to add progress prints
  import collections
  advantage_losses = collections.defaultdict(list)
  
  for i in range(FLAGS.num_iterations):
    if FLAGS.verbose:
      logging.info("Iteration %d/%d", i + 1, FLAGS.num_iterations)
      
    for p in range(game.num_players()):
      # logging.info("  Player %d traversals...", p)
      for t in range(FLAGS.num_traversals):
        deep_cfr_solver._traverse_game_tree(deep_cfr_solver._root_node, p)
        if (t + 1) % 100 == 0 and FLAGS.verbose:
             logging.info("    Player %d: Traversal %d/%d", p, t + 1, FLAGS.num_traversals)
      
      if deep_cfr_solver._reinitialize_advantage_networks:
        # Re-initialize advantage network for player and train from scratch.
        deep_cfr_solver.reinitialize_advantage_network(p)
      
      # Re-initialize advantage networks and train from scratch.
      loss = deep_cfr_solver._learn_advantage_network(p)
      advantage_losses[p].append(loss)
      if FLAGS.verbose:
          logging.info("  Player %d advantage loss: %s", p, str(loss))
          
    deep_cfr_solver._iteration += 1
  
  # Train policy network.
  logging.info("Training policy network...")
  policy_loss = deep_cfr_solver._learn_strategy_network()
  logging.info("Policy loss: %s", str(policy_loss))
  
  elapsed = time.time() - start_time
  logging.info("\nTraining complete!")
  logging.info("  Total time: %.1f minutes", elapsed / 60)
  
  # Print training statistics
  for player, losses in advantage_losses.items():
    # Filter out None values
    valid_losses = [x for x in losses if x is not None]
    if FLAGS.verbose and len(valid_losses) > 0:
      logging.info("Advantage losses for player %d:", player)
      logging.info("  First: [%s]", ", ".join(f"{x:.4f}" for x in valid_losses[:3]))
      logging.info("  Last:  [%s]", ", ".join(f"{x:.4f}" for x in valid_losses[-3:]))
    logging.info("  Advantage buffer size for player %d: %d",
                 player, len(deep_cfr_solver.advantage_buffers[player]))
  
  logging.info("Strategy buffer size: %d",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: %.6f", policy_loss)
  
  # Save the trained policy
  logging.info("\nSaving policy...")
  policy_path = os.path.join(FLAGS.output_dir, "deep_cfr_policy.pkl")
  
  # Create a callable policy object
  truco_policy = TrucoPolicy(deep_cfr_solver)
  
  with open(policy_path, 'wb') as f:
    pickle.dump({
        'solver': deep_cfr_solver,
        'policy': truco_policy,
        'num_iterations': FLAGS.num_iterations,
        'num_traversals': FLAGS.num_traversals,
        'final_policy_loss': policy_loss,
        'training_time': elapsed
    }, f)
  
  logging.info("Policy saved to: %s", policy_path)
  
  # Try to compute exploitability (may be too expensive for large games)
  if FLAGS.compute_nash_conv:
    try:
      logging.info("\nComputing exploitability (this may take a while)...")
      average_policy = policy.tabular_policy_from_callable(
          game, truco_policy)
      nash_conv = exploitability.nash_conv(game, average_policy)
      logging.info("NashConv: %.6f", nash_conv)
      
      with open(policy_path, 'wb') as f:
        pickle.dump({
            'solver': deep_cfr_solver,
            'policy': truco_policy,
            'num_iterations': FLAGS.num_iterations,
            'num_traversals': FLAGS.num_traversals,
            'final_policy_loss': policy_loss,
            'nash_conv': nash_conv,
            'training_time': elapsed
        }, f)
    except Exception as e:
      logging.warning("Could not compute exploitability: %s", e)
      logging.info("This is expected for large games like Truco.")
  else:
    logging.info("\nSkipping exploitability calculation (use --compute_nash_conv to enable).")
  
  logging.info("\n" + "="*60)
  logging.info("TRAINING SUMMARY")
  logging.info("="*60)
  logging.info("Game:              Truco")
  logging.info("Iterations:        %d", FLAGS.num_iterations)
  logging.info("Traversals/iter:   %d", FLAGS.num_traversals)
  logging.info("Total time:        %.1f minutes", elapsed / 60)
  logging.info("Policy loss:       %.6f", policy_loss)
  logging.info("Output:            %s", FLAGS.output_dir)
  logging.info("="*60)
  
  logging.info("\nNext steps:")
  logging.info("  1. Evaluate with: python truco_eval_policies.py \\")
  logging.info("       --policy_path=%s", policy_path)
  logging.info("  2. Play against: python truco_play_interactive.py \\")
  logging.info("       --opponent=policy --policy_path=%s", policy_path)


if __name__ == "__main__":
  app.run(main)
