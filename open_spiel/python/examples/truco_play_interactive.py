#!/usr/bin/env python3
# Copyright 2025 DeepMind Technologies Limited
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

"""Interactive play against AI agents in Truco.

This script allows you to play Truco against various AI opponents.
"""

import pickle
import numpy as np
from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import ismcts
from open_spiel.python.bots import human
from open_spiel.python.bots import policy as policy_bot
from open_spiel.python.bots import uniform_random
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_enum("opponent", "random", 
                 ["random", "ismcts", "policy"],
                 "Type of opponent to play against")
flags.DEFINE_string("policy_path", None, 
                   "Path to policy file (required if opponent=policy)")
flags.DEFINE_integer("ismcts_simulations", 500, "IS-MCTS simulations per move")
flags.DEFINE_integer("max_world_samples", 3, "IS-MCTS world samples")
flags.DEFINE_float("uct_c", 1.5, "UCT exploration constant")
flags.DEFINE_integer("seed", None, "Random seed (None for random)")
flags.DEFINE_bool("human_first", True, "Human plays as player 0")


class TrucoPolicy:
  """A policy that uses the Deep CFR solver."""
  def __init__(self, solver):
    self.solver = solver

  def __call__(self, state):
    return self.solver.action_probabilities(state)

  def action_probabilities(self, state, player_id=None):
    # Get raw probabilities from the solver
    probs = self.solver.action_probabilities(state, player_id)
    
    # Get legal actions
    legal_actions = state.legal_actions(player_id if player_id is not None else state.current_player())
    
    # print(f"DEBUG: Legal actions: {legal_actions}")
    # print(f"DEBUG: Raw probs from solver: {probs}")

    # Filter for legal actions
    legal_probs_map = {a: probs.get(a, 0.0) for a in legal_actions}
    
    # Check for NaNs or negatives
    clean_probs = []
    for a in legal_actions:
        p = legal_probs_map[a]
        if np.isnan(p) or p < 0:
            print(f"DEBUG: Invalid probability {p} for action {a}. Treating as 0.")
            p = 0.0
        clean_probs.append(p)
    
    total = sum(clean_probs)
    # print(f"DEBUG: Total probability mass: {total}")

    if total <= 1e-9:
        print("DEBUG: Total probability is effectively zero. Using uniform distribution.")
        return {a: 1.0 / len(legal_actions) for a in legal_actions}
    
    # Normalize using numpy for precision
    normalized_values = np.array(clean_probs, dtype=np.float64)
    normalized_values = normalized_values / normalized_values.sum()
    
    # print(f"DEBUG: Normalized values: {normalized_values}")
    # print(f"DEBUG: Sum: {normalized_values.sum()}")

    return dict(zip(legal_actions, normalized_values))


def create_opponent(game, opponent_type, player_id, rng):
  """Create an AI opponent."""
  if opponent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif opponent_type == "ismcts":
    return ismcts.ISMCTSBot(
        FLAGS.seed,
        FLAGS.ismcts_simulations,
        FLAGS.max_world_samples,
        FLAGS.uct_c,
        random_state=rng)
  elif opponent_type == "policy":
    if FLAGS.policy_path is None:
      raise ValueError("Must specify --policy_path when opponent=policy")
    with open(FLAGS.policy_path, 'rb') as f:
      policy_data = pickle.load(f)
    policy = policy_data['policy']
    return policy_bot.PolicyBot(player_id, rng, policy)
  else:
    raise ValueError(f"Unknown opponent type: {opponent_type}")


def main(_):
  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)
  
  logging.info("Loading Truco game...")
  game = pyspiel.load_game("truco")
  
  rng = np.random.RandomState(FLAGS.seed)
  
  # Create players
  human_player_id = 0 if FLAGS.human_first else 1
  ai_player_id = 1 - human_player_id
  
  human_bot = human.HumanBot()
  ai_bot = create_opponent(game, FLAGS.opponent, ai_player_id, rng)
  
  bots = [None, None]
  bots[human_player_id] = human_bot
  bots[ai_player_id] = ai_bot
  
  logging.info("\n" + "="*60)
  logging.info("Starting Truco game!")
  logging.info(f"You are Player {human_player_id}")
  logging.info(f"Opponent: {FLAGS.opponent.upper()}")
  logging.info("="*60 + "\n")
  
  # Play game
  state = game.new_initial_state()
  
  while not state.is_terminal():
    print(f"\n{state}\n")
    
    current_player = state.current_player()
    
    if state.is_chance_node():
      # Chance node
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = np.random.choice(action_list, p=prob_list)
      print(f"Dealing cards...")
    else:
      # Player decision
      if current_player == human_player_id:
        print(f"\nYour turn (Player {current_player}):")
      else:
        print(f"\nOpponent's turn (Player {current_player}):")
      
      action = bots[current_player].step(state)
      action_str = state.action_to_string(current_player, action)
      
      if current_player == human_player_id:
        print(f"You played: {action_str}")
      else:
        print(f"Opponent played: {action_str}")
      
      # Inform other bot
      for i, bot in enumerate(bots):
        if i != current_player:
          bot.inform_action(state, current_player, action)
    
    state.apply_action(action)
  
  # Game over
  print(f"\n{state}\n")
  print("="*60)
  print("Game Over!")
  print("="*60)
  
  returns = state.returns()
  if returns[human_player_id] > returns[ai_player_id]:
    print("🎉 You won!")
  elif returns[ai_player_id] > returns[human_player_id]:
    print("😞 You lost!")
  else:
    print("🤝 Draw!")
  
  print(f"\nFinal returns:")
  print(f"  You (Player {human_player_id}): {returns[human_player_id]:.1f}")
  print(f"  Opponent (Player {ai_player_id}): {returns[ai_player_id]:.1f}")


if __name__ == "__main__":
  app.run(main)
