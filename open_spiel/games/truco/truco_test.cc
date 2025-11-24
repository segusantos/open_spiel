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
std::unique_ptr<State> DealFixedHand(const std::shared_ptr<const Game>& game,
                                     const std::vector<Action>& deal) {
  std::unique_ptr<State> state = game->NewInitialState();
  for (Action card : deal) {
    state->ApplyAction(card);
  }
  return state;
}

void FullGameToThirtyTest() {
  auto game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();

  int hands = 0;
  while (!state->IsTerminal() && hands < 1000) {
    if (state->IsChanceNode()) {
      state->ApplyAction(state->LegalActions()[0]);
    } else if (state->LegalActions()[0] == kNewHandAction) {
      state->ApplyAction(kNewHandAction);
      hands++;
    } else {
      state->ApplyAction(state->LegalActions()[0]);
    }
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());
  // With the new absolute scoring, returns are zero-sum (margin).
  // The winner should have a positive return, but it might not be exactly 30
  // if the opponent scored points (e.g. 30-29 win -> return 1).
  // However, since this test plays randomly/deterministically (first action),
  // we should check that the game ended correctly.
  // We can't easily access game_points_ here without casting, but we can check
  // that returns are non-zero and sum to 0.
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_TRUE(std::abs(returns[0]) > 0);
}

void AbsoluteScoreTest() {
  auto game = LoadGame("truco");
  // P0: 1 Espada (20), 7 Oro (36), 3 Basto (2)
  // P1: 7 Espada (26), 6 Espada (25), 5 Espada (24)
  auto state = DealFixedHand(game, {20, 26, 36, 25, 2, 24});

  // P0 calls Truco
  state->ApplyAction(kRaiseTrucoAction);
  // P1 accepts
  state->ApplyAction(kAcceptBetAction);

  // P0 plays 1 Espada (wins)
  state->ApplyAction(20);
  state->ApplyAction(26);

  // P0 plays 7 Oro (wins)
  state->ApplyAction(36);
  state->ApplyAction(25);

  // P0 wins hand (2 points for Truco)
  state->ApplyAction(kNewHandAction);

  // Check returns (should be +2, -2)
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 2);
  SPIEL_CHECK_EQ(returns[1], -2);

  // Check string representation for absolute score
  // Should be "| Score: P0 [ 2 ]  vs  P1 [ 0 ]"
  std::string str = state->ToString();
  SPIEL_CHECK_NE(str.find("| Score: P0 [ 2 ]  vs  P1 [ 0 ]"),
                 std::string::npos);

  // Now let P1 win a hand worth 1 point
  // Deal next hand...
  // We can't easily force the deal in the same state object without complex
  // setup, but we can verify the score didn't go to -2 for P1 in the string.
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
  // Game should be waiting for NewHand action
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->LegalActions()[0], kNewHandAction);
  state->ApplyAction(kNewHandAction);
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
  state->ApplyAction(kNewHandAction);
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
  state->ApplyAction(kNewHandAction);
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P1 won envido decline (3 pts) + hand (1 pt) = 4 pts total
  SPIEL_CHECK_EQ(returns[0], -4);
  SPIEL_CHECK_EQ(returns[1], 4);
}

void TrucoDeclineTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kRejectBetAction);
  // Truco decline ends the hand immediately, but game continues to 30 pts
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void TrucoValeCuatroTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 10, 21, 11, 22, 12});
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(20);
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(10);
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(21);
  state->ApplyAction(11);
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P0 won hand with Vale Cuatro accepted = 4 pts
  SPIEL_CHECK_EQ(returns[0], 4);
  SPIEL_CHECK_EQ(returns[1], -4);
}

void TrucoImmediateRetrucoTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(0);
  state->ApplyAction(10);
  state->ApplyAction(1);
  state->ApplyAction(11);
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // P0 won hand with Retruco accepted = 3 pts
  SPIEL_CHECK_EQ(returns[0], 3);
  SPIEL_CHECK_EQ(returns[1], -3);
}

void TrucoSecondTrickTurnOrderTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  std::vector<Action> actions = {kEnvidoAction, kRealEnvidoAction,
                                 kRejectBetAction, 3, 20};
  for (Action action : actions) {
    state->ApplyActionWithLegalityCheck(action);
  }
  
  auto legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRejectBetAction) == legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(), 21) !=
                   legal_actions.end());
}

void TrucoSecondTrickCloneTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  std::vector<Action> actions = {kEnvidoAction, kRealEnvidoAction,
                                 kRejectBetAction, 3, 20};
  for (Action action : actions) {
    std::unique_ptr<State> clone = state->Clone();
    state->ApplyActionWithLegalityCheck(action);
    clone->ApplyActionWithLegalityCheck(action);
    SPIEL_CHECK_EQ(state->ToString(), clone->ToString());
  }
  auto legal_actions = state->LegalActions();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRejectBetAction) == legal_actions.end());
}

void FaltaEnvidoAcceptTest() {
  auto game = LoadGame("truco");
  // P0: 5,6,7 de Oro (envido: 20+7+6=33)
  // P1: 1,2,3 de Basto (envido: 20+3+2=25)
  auto state = DealFixedHand(game, {34, 0, 35, 1, 36, 2});
  state->ApplyAction(kFaltaEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  // According to the Wikipedia rule implemented: both teams are in "malas"
  // (scores 0-14) so Falta Envido at the start awards enough points to win
  // the partido (reach kTargetScore). The winner should immediately get
  // the points required to reach kTargetScore and the game becomes terminal.
  const auto returns_after_envido = state->Returns();
  SPIEL_CHECK_EQ(returns_after_envido[0], kTargetScore);
  SPIEL_CHECK_EQ(returns_after_envido[1], -kTargetScore);
  SPIEL_CHECK_TRUE(state->IsTerminal());
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
  if (!state->IsChanceNode() && state->LegalActions()[0] != kNewHandAction) {
    state->ApplyAction(35);  // P0: 6 de Oro
    state->ApplyAction(5);   // P1: 6 de Basto (tie)
  }
  // Game is multi-hand (plays to 30 points), so NOT terminal after one hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Dealing for next hand
  const auto returns = state->Returns();
  // Mano (P0) wins when all tricks are tied, gets 1 point
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void EnvidoFirstTest() {
  auto game = LoadGame("truco");
  // P0: 1 Espada (20), 7 Oro (36), 3 Basto (2) -> Envido 20+0=20 (bad)
  // P1: 7 Espada (26), 6 Espada (25), 5 Espada (24) -> Envido 20+7+6=33 (good)
  auto state = DealFixedHand(game, {20, 26, 36, 25, 2, 24});

  // P0 calls Truco
  state->ApplyAction(kRaiseTrucoAction);

  // P1 calls Envido (Envido First)
  state->ApplyAction(kEnvidoAction);

  // P0 accepts Envido
  state->ApplyAction(kAcceptBetAction);

  // Envido should be resolved (P1 wins 2 points)
  auto rewards = state->Rewards();
  SPIEL_CHECK_EQ(rewards[0], -2);
  SPIEL_CHECK_EQ(rewards[1], 2);

  // Now P1 must answer Truco
  // P1 accepts Truco
  state->ApplyAction(kAcceptBetAction);

  // Game continues
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void TieFirstTrickWinnerSecondTakesHandTest() {
  auto game = LoadGame("truco");
  // P0: 4 Oro (33), 1 Espada (20), ...
  // P1: 4 Basto (3), 4 Copa (13), ...
  // Deal: P0, P1, P0, P1, P0, P1
  auto state = DealFixedHand(game, {33, 3, 20, 13, 35, 14});

  state->ApplyAction(33);  // P0 plays 4 Oro
  state->ApplyAction(3);   // P1 plays 4 Basto (Tie)

  // Mano (P0) should lead second trick after tie
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  state->ApplyAction(20);  // P0 plays 1 Espada
  state->ApplyAction(13);  // P1 plays 4 Copa (P0 wins trick)

  // P0 won 2nd trick after 1st was tied -> P0 wins hand
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void FirstWonSecondTiedWinnerFirstTakesHandTest() {
  auto game = LoadGame("truco");
  // P0: 1 Espada (20), 4 Oro (33), ...
  // P1: 4 Copa (13), 4 Basto (3), ...
  // Deal: P0, P1, P0, P1, P0, P1
  auto state = DealFixedHand(game, {20, 13, 33, 3, 35, 14});

  state->ApplyAction(20);  // P0 plays 1 Espada
  state->ApplyAction(13);  // P1 plays 4 Copa (P0 wins)

  state->ApplyAction(33);  // P0 leads 4 Oro
  state->ApplyAction(3);   // P1 plays 4 Basto (Tie)

  // P0 won 1st, 2nd tied -> P0 wins hand immediately
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], 1);
  SPIEL_CHECK_EQ(returns[1], -1);
}

void RetrucoDeclinePointsTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});

  state->ApplyAction(kRaiseTrucoAction);  // Truco (P0)
  state->ApplyAction(kRaiseTrucoAction);  // Retruco (P1)
  state->ApplyAction(kRejectBetAction);   // Reject (P0)

  // Rejecting Retruco gives 2 points to the caller (P1)
  SPIEL_CHECK_FALSE(state->IsTerminal());
  state->ApplyAction(kNewHandAction);
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0], -2);
  SPIEL_CHECK_EQ(returns[1], 2);
}

void EnvidoIllegalInSecondTrickTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});

  // Play first trick
  state->ApplyAction(0);
  state->ApplyAction(10);

  // Now in second trick
  auto legal_actions = state->LegalActions();

  // Envido actions should NOT be legal anymore
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kEnvidoAction) == legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRealEnvidoAction) == legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kFaltaEnvidoAction) == legal_actions.end());

  // Truco should still be legal
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRaiseTrucoAction) != legal_actions.end());
}

void TensorShapeTest() {
  std::shared_ptr<const Game> game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();
  
  // Check Observation Tensor Shape
  std::vector<int> obs_shape = game->ObservationTensorShape();
  int expected_size = 0;
  // Player: 2
  expected_size += 2;
  // Hand: 40
  expected_size += 40;
  // Tricks: 3 * 2 * 40 = 240
  expected_size += 240;
  // Trick Info: 3 * (2+1) + 3 * 2 = 9 + 6 = 15
  expected_size += 15;
  // Game Score: 2
  expected_size += 2;
  // Truco Level: 4
  expected_size += 4;
  // Envido Sequence: 4 * 3 = 12
  expected_size += 12;
  // Envido State: 2
  expected_size += 2;
  // Envido Score: 2
  expected_size += 2;
  // Mano: 2
  expected_size += 2;
  
  int calculated_size = obs_shape[0];
  SPIEL_CHECK_EQ(calculated_size, expected_size);
}

void TrickAnalysisTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();
  
  // Deal cards
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  
  // P0 plays card 0
  state->ApplyAction(state->LegalActions()[0]);
  // P1 plays card 0 (assuming different card)
  state->ApplyAction(state->LegalActions()[0]);
  
  // Trick 1 finished.
  
  std::vector<float> tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(tensor));
  
  // Verify that some bits in the trick analysis section are set
  // We need to calculate the offset
  int offset = 2 + 40 + 240; // Player + Hand + Tricks
  
  // Trick Winners (3 * 3 = 9 bits)
  // Trick Leaders (3 * 2 = 6 bits)
  
  bool found_winner = false;
  for (int i = 0; i < 9; ++i) {
    if (tensor[offset + i] == 1.0) found_winner = true;
  }
  
  bool found_leader = false;
  for (int i = 0; i < 6; ++i) {
    if (tensor[offset + 9 + i] == 1.0) found_leader = true;
  }
  
  SPIEL_CHECK_TRUE(found_winner);
  SPIEL_CHECK_TRUE(found_leader);
}

void TieFirstTieSecondThirdDecidesTest() {
  auto game = LoadGame("truco");
  // P0 (Mano): 4 Oro (33), 5 Oro (34), 1 Espada (20)
  // P1: 4 Basto (3), 5 Basto (4), 1 Basto (13)
  // 1st trick: 4 Oro vs 4 Basto (Tie)
  // 2nd trick: 5 Oro vs 5 Basto (Tie)
  // 3rd trick: 1 Espada (14) vs 1 Basto (13) -> P0 wins
  // Wait, if P0 wins, Mano wins anyway.
  // Let's make P1 win the 3rd trick.
  // P0: 4 Oro (33), 5 Oro (34), 1 Basto (13)
  // P1: 4 Basto (3), 5 Basto (4), 1 Espada (20)
  auto state = DealFixedHand(game, {33, 3, 34, 4, 13, 20});

  state->ApplyAction(33);  // P0: 4 Oro
  state->ApplyAction(3);   // P1: 4 Basto (Tie)

  state->ApplyAction(34);  // P0: 5 Oro
  state->ApplyAction(4);   // P1: 5 Basto (Tie)

  // If bug exists, hand ends here and Mano (P0) wins.
  // If correct, game continues to 3rd trick.

  if (state->IsChanceNode() || state->LegalActions()[0] == kNewHandAction) {
    // Hand ended early
    const auto returns = state->Returns();
    if (returns[0] > 0) {
      SPIEL_CHECK_TRUE(
          false &&
          "Hand ended early after 2 ties, Mano won. Should play 3rd trick.");
    }
  } else {
    // Play 3rd trick
    state->ApplyAction(13);  // P0: 1 Basto
    state->ApplyAction(20);  // P1: 1 Espada (Wins trick and hand)

    state->ApplyAction(kNewHandAction);
    const auto returns = state->Returns();
    SPIEL_CHECK_EQ(returns[0], -1);
    SPIEL_CHECK_EQ(returns[1], 1);
  }
}

void ThirdTrickLeaderTest() {
  auto game = LoadGame("truco");
  // Deal: P0 gets 20 (1 Espada), 3 (4 Basto), 1 (4 Copa). 
  //       P1 gets 13 (1 Basto), 10 (1 Copa), 26 (7 Espada).
  auto state = DealFixedHand(game, {20, 13, 3, 10, 1, 26});
  
  // Trick 1
  state->ApplyAction(20); // P0: 1 Espada
  state->ApplyAction(13); // P1: 1 Basto
  // P0 wins (14 > 13).
  
  // Trick 2
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0); // P0 leads
  state->ApplyAction(3);  // P0: 4 Basto
  state->ApplyAction(26); // P1: 7 Espada
  // P1 wins (12 > 1).
  
  // Trick 3
  // P1 should lead because they won the previous trick.
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
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
  open_spiel::truco::TrucoSecondTrickTurnOrderTest();
  open_spiel::truco::TrucoSecondTrickCloneTest();
  open_spiel::truco::FaltaEnvidoAcceptTest();
  open_spiel::truco::EnvidoTieGoesToManoTest();
  open_spiel::truco::AllTricksTiedManoWinsTest();
  open_spiel::truco::EnvidoFirstTest();
  open_spiel::truco::FullGameToThirtyTest();
  open_spiel::truco::AbsoluteScoreTest();
  open_spiel::truco::TensorShapeTest();
  open_spiel::truco::TrickAnalysisTensorTest();

  // New tests
  open_spiel::truco::TieFirstTrickWinnerSecondTakesHandTest();
  open_spiel::truco::FirstWonSecondTiedWinnerFirstTakesHandTest();
  open_spiel::truco::RetrucoDeclinePointsTest();
  open_spiel::truco::EnvidoIllegalInSecondTrickTest();
  open_spiel::truco::ThirdTrickLeaderTest();
  open_spiel::truco::TieFirstTieSecondThirdDecidesTest();
}
