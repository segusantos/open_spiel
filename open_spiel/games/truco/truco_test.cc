// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/truco/truco.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace truco {
namespace {

namespace testing = open_spiel::testing;

void BasicTrucoTests() {
  testing::LoadGameTest("truco");
  std::shared_ptr<const Game> game = LoadGame("truco");
  testing::ChanceOutcomesTest(*game);
  open_spiel::testing::RandomSimTest(*game, 100);
  open_spiel::testing::ResampleInfostateTest(*game, 100);
  auto observer = game->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
}
void FullGameToThirtyTest() {
  auto game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();
  // Simulate 30 hands, P0 always wins (1 point per hand)
  for (int i = 0; i < 30; ++i) {
    // Deal cards (simulate chance node)
    while (state->IsChanceNode()) {
      state->ApplyAction(state->LegalActions()[0]);
    }
    // P0 plays first legal card, P1 plays next, repeat for two tricks
    for (int t = 0; t < 2; ++t) {
      state->ApplyAction(state->LegalActions()[0]);
      state->ApplyAction(state->LegalActions()[0]);
    }
    // After each hand, game should not be terminal until the last
    if (i < 29) {
      SPIEL_CHECK_FALSE(state->IsTerminal());
    }
  }
  // Now, after 30 points, the game should be terminal
  SPIEL_CHECK_TRUE(state->IsTerminal());
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 30);
  SPIEL_CHECK_EQ(returns[1], -30);
}

void StartingPlayerAndDeterministicPlayTest() {
  std::shared_ptr<const Game> game =
      LoadGame("truco", {{"starting_player", GameParameter(1)}});
  std::unique_ptr<State> state = game->NewInitialState();

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  // Deal: P0 gets 0,1,2 and P1 gets 20,21,22
  const std::vector<Action> dealing = {0, 20, 1, 21, 2, 22};
  for (Action card : dealing) {
    state->ApplyAction(card);
  }

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  state->ApplyAction(20);  // Player 1 plays Ancho de Espada.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(0);  // Player 0 plays Ancho de Basto.

  // Ancho de Espada (14) beats Ancho de Basto (13), so P1 wins and leads
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  state->ApplyAction(21);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(1);

  // Game is now multi-hand (plays to 30 points), so after one hand it should
  // NOT be terminal
  SPIEL_CHECK_FALSE(state->IsTerminal());
  // Game should be dealing cards for Hand 1
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const std::vector<double> returns = state->Returns();
  // P1 wins both tricks (Ancho de Espada > Ancho de Basto in trick 1,
  // and wins trick 2 as well), so game score is P0=-1, P1=1
  SPIEL_CHECK_EQ(returns[0], -1.0);
  SPIEL_CHECK_EQ(returns[1], 1.0);

  const std::string info_string = state->InformationStateString(0);
  SPIEL_CHECK_NE(info_string.find("H:"), std::string::npos);
}

void ObserverTensorCoverageTest() {
  std::shared_ptr<const Game> game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }

  const Player player = state->CurrentPlayer();
  std::vector<float> obs_tensor(game->ObservationTensorSize());
  state->ObservationTensor(player, absl::MakeSpan(obs_tensor));
  float obs_sum = std::accumulate(obs_tensor.begin(), obs_tensor.end(), 0.0f);
  SPIEL_CHECK_GT(obs_sum, 0.0f);

  std::vector<float> info_tensor(game->InformationStateTensorSize());
  state->InformationStateTensor(player, absl::MakeSpan(info_tensor));
  float info_sum =
      std::accumulate(info_tensor.begin(), info_tensor.end(), 0.0f);
  SPIEL_CHECK_GT(info_sum, 0.0f);
}

std::unique_ptr<State> DealFixedHand(const std::shared_ptr<const Game>& game,
                                     const std::vector<Action>& deal) {
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action card : deal) {
    state->ApplyAction(card);
  }
  return state;
}

void EnvidoAcceptTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(0);
  state->ApplyAction(10);
  state->ApplyAction(1);
  state->ApplyAction(11);
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P0 won envido (2 pts) + hand (1 pt) = 3 pts total
  SPIEL_CHECK_EQ(returns[0], 3);
  SPIEL_CHECK_EQ(returns[1], -3);
}

void EnvidoRaiseDeclineTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kRealEnvidoAction);
  state->ApplyAction(kRejectBetAction);
  const auto after_decline = state->Returns();
  SPIEL_CHECK_EQ(after_decline[0], -3);
  SPIEL_CHECK_EQ(after_decline[1], 3);
  state->ApplyAction(3);   // P0 plays card 3 (Ancho de Copa)
  state->ApplyAction(20);  // P1 plays card 20 (Ancho de Espada) - P1 wins trick
  state->ApplyAction(21);  // P1 leads trick 2 with card 21
  state->ApplyAction(4);   // P0 plays card 4
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P1 won envido decline (3 pts) + hand (1 pt) = 4 pts total
  SPIEL_CHECK_EQ(returns[0], -4);
  SPIEL_CHECK_EQ(returns[1], 4);
}

void TrucoDeclineTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kTrucoAction);
  state->ApplyAction(kRejectBetAction);
  // Truco decline ends the hand immediately, but game continues to 30 pts
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void TrucoValeCuatroTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 10, 21, 11, 22, 12});
  state->ApplyAction(kTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(20);
  state->ApplyAction(kRetrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(10);
  state->ApplyAction(kValeCuatroAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(21);
  state->ApplyAction(11);
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P0 won hand with Vale Cuatro accepted = 4 pts
  SPIEL_CHECK_EQ(returns[0], 4);
  SPIEL_CHECK_EQ(returns[1], -4);
}

void TrucoImmediateRetrucoTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kTrucoAction);
  state->ApplyAction(kRetrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(0);
  state->ApplyAction(10);
  state->ApplyAction(1);
  state->ApplyAction(11);
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P0 won hand with Retruco accepted = 3 pts
  SPIEL_CHECK_EQ(returns[0], 3);
  SPIEL_CHECK_EQ(returns[1], -3);
}

void TrucoCloneRegressionTest() {
  auto game = LoadGame("truco");
  // Deal the deterministic hand that previously triggered a clone mismatch
  // during the random simulation test (see #action trace in the logs).
  auto state = DealFixedHand(game, {29, 16, 17, 4, 0, 12});
  std::vector<Action> actions = {29, 12, 41, 46, 43, 46, 16, 44, 46, 0, 17, 4};
  for (Action action : actions) {
    std::unique_ptr<State> clone = state->Clone();
    clone->ApplyActionWithLegalityCheck(action);
    state->ApplyActionWithLegalityCheck(action);
  }
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
}

void TrucoSecondTrickTurnOrderTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  std::vector<Action> actions = {40, 41, 47, 3, 20};
  for (Action action : actions) {
    state->ApplyActionWithLegalityCheck(action);
  }
  std::cout << "After actions CurrentPlayer=" << state->CurrentPlayer()
            << std::endl;
  std::cout << "Legal actions:";
  auto legal_actions = state->LegalActions();
  for (Action action : legal_actions) {
    std::cout << " " << action << " ("
              << state->ActionToString(state->CurrentPlayer(), action) << ")";
  }
  std::cout << std::endl;
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(), 4) ==
                   legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(), 21) !=
                   legal_actions.end());
}

void TrucoSecondTrickCloneTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  std::vector<Action> actions = {40, 41, 47, 3, 20};
  for (Action action : actions) {
    std::unique_ptr<State> clone = state->Clone();
    state->ApplyActionWithLegalityCheck(action);
    clone->ApplyActionWithLegalityCheck(action);
    SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  }
  auto legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(), 4) ==
                   legal_actions.end());
}

void FaltaEnvidoAcceptTest() {
  auto game = LoadGame("truco");
  // P0: 5,6,7 de Oro (envido: 20+7+6=33)
  // P1: 1,2,3 de Basto (envido: 20+3+2=25)
  auto state = DealFixedHand(game, {34, 0, 35, 1, 36, 2});
  state->ApplyAction(kFaltaEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  // P0 should win with 33 vs 25. Falta Envido at start: P1 in malas (0 pts),
  // so P0 gets (15-0) = 15 points
  const auto returns_after_envido = state->Returns();
  SPIEL_CHECK_EQ(returns_after_envido[0], 15);
  SPIEL_CHECK_EQ(returns_after_envido[1], -15);
  // Hand should continue
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(34);  // P0 plays 5 de Oro
  state->ApplyAction(0);   // P1 plays 1 de Basto (Ancho de Basto wins)
  state->ApplyAction(1);   // P1 leads with 2 de Basto
  state->ApplyAction(35);  // P0 plays 6 de Oro
  // After one hand, game should NOT be terminal (plays to 30)
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto final_returns = state->Returns();
  // P1 won 2 tricks so gets 1 point for the hand, total: 15-1=14
  SPIEL_CHECK_EQ(final_returns[0], 14);
  SPIEL_CHECK_EQ(final_returns[1], -14);
}

void EnvidoTieGoesToManoTest() {
  auto game = LoadGame("truco");
  // Both players have envido 7 (no matching suits, single 7 highest card)
  // P0 (mano): 7 de Oro (36), 5 de Basto (4), 4 de Copa (13)
  // P1: 7 de Espada (26), 6 de Basto (5), 3 de Copa (12)
  auto state = DealFixedHand(game, {36, 26, 4, 5, 13, 12});
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  // P0 (mano) should win on tie (both have 7), gets 2 points for Envido
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 2);
  SPIEL_CHECK_EQ(returns[1], -2);
  // Hand should still be playable (not terminal after envido resolution)
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void AllTricksTiedManoWinsTest() {
  auto game = LoadGame("truco");
  // Deal cards where all tricks tie
  // P0 (mano): 4,5,6 de Oro (33,34,35)
  // P1: 4,5,6 de Basto (3,4,5)  (same rank = tie)
  auto state = DealFixedHand(game, {33, 3, 34, 4, 35, 5});
  state->ApplyAction(33);  // P0: 4 de Oro
  state->ApplyAction(3);   // P1: 4 de Basto (tie - mano leads)
  // After tied first trick, mano (P0) should lead
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(34);  // P0: 5 de Oro
  state->ApplyAction(4);   // P1: 5 de Basto (tie - mano leads again)
  // After 2 tied tricks, game may already be decided
  if (!state->IsChanceNode()) {
    state->ApplyAction(35);  // P0: 6 de Oro
    state->ApplyAction(5);   // P1: 6 de Basto (tie)
  }
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // Mano (P0) wins when all tricks are tied, gets 1 point
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

}  // namespace
}  // namespace truco
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::truco::BasicTrucoTests();
  open_spiel::truco::StartingPlayerAndDeterministicPlayTest();
  open_spiel::truco::ObserverTensorCoverageTest();
  open_spiel::truco::EnvidoAcceptTest();
  open_spiel::truco::EnvidoRaiseDeclineTest();
  open_spiel::truco::TrucoDeclineTest();
  open_spiel::truco::TrucoValeCuatroTest();
  open_spiel::truco::TrucoImmediateRetrucoTest();
  open_spiel::truco::TrucoCloneRegressionTest();
  open_spiel::truco::TrucoSecondTrickTurnOrderTest();
  open_spiel::truco::TrucoSecondTrickCloneTest();
  open_spiel::truco::FaltaEnvidoAcceptTest();
  open_spiel::truco::EnvidoTieGoesToManoTest();
  open_spiel::truco::AllTricksTiedManoWinsTest();
  open_spiel::truco::FullGameToThirtyTest();
}
