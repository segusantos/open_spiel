// Copyright 2025 DeepMind Technologies Limited
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
#include <cmath>
#include <memory>
#include <numeric>
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

// Helper to compute the expected potential for a given score difference
double ExpectedPotential(int my_score, int opp_score) {
  return static_cast<double>(my_score - opp_score) /
         static_cast<double>(kTargetScore);
}

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
    } else {
      state->ApplyAction(state->LegalActions()[0]);
    }
  }

  SPIEL_CHECK_TRUE(state->IsTerminal());
  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_TRUE(std::abs(returns[0]) == 1.0);
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

  auto returns = state->Returns();
  // Returns should be zero-sum and positive for P0
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);

  // Check string representation for absolute score
  // Should be "| Score: P0 [ 2 ]  vs  P1 [ 0 ]"
  std::string str = state->ToString();
  SPIEL_CHECK_NE(str.find("| Score: P0 [ 2 ]  vs  P1 [ 0 ]"),
                 std::string::npos);
}

void StartingPlayerAndDeterministicPlayTest() {
  std::shared_ptr<const Game> game =
      LoadGame("truco", {{"starting_player", GameParameter(1)}});
  std::unique_ptr<State> state = game->NewInitialState();

  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const std::vector<Action> dealing = {0, 20, 1, 21, 2, 22};
  for (Action card : dealing) {
    state->ApplyAction(card);
  }

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);

  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(20);

  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  state->ApplyAction(21);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  state->ApplyAction(1);

  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);

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
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
}

void EnvidoRaiseDeclineTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {3, 20, 4, 21, 5, 22});
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kRealEnvidoAction);
  state->ApplyAction(kRejectBetAction);
  const auto after_decline = state->Returns();
  SPIEL_CHECK_EQ(after_decline[0] + after_decline[1], 0);
  SPIEL_CHECK_LT(after_decline[0], 0);
  state->ApplyAction(3);
  state->ApplyAction(20);
  state->ApplyAction(21);
  state->ApplyAction(4);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_LT(returns[0], 0);
}

void TrucoDeclineTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kRejectBetAction);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
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
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
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
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
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
  auto state = DealFixedHand(game, {34, 0, 35, 1, 36, 2});
  state->ApplyAction(kFaltaEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  const auto returns_after_envido = state->Returns();
  SPIEL_CHECK_EQ(returns_after_envido[0], 1.0);
  SPIEL_CHECK_EQ(returns_after_envido[1], -1.0);
  SPIEL_CHECK_TRUE(state->IsTerminal());
}

void EnvidoTieGoesToManoTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {36, 26, 4, 5, 13, 12});
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kAcceptBetAction);
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void AllTricksTiedManoWinsTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {33, 3, 34, 4, 35, 5});
  state->ApplyAction(33);
  state->ApplyAction(3);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(34);
  state->ApplyAction(4);
  state->ApplyAction(35);
  state->ApplyAction(5);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
}

void EnvidoFirstTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 26, 36, 25, 2, 24});

  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kAcceptBetAction);

  auto rewards = state->Rewards();
  SPIEL_CHECK_EQ(rewards[0] + rewards[1], 0);
  SPIEL_CHECK_LT(rewards[0], 0);

  state->ApplyAction(kAcceptBetAction);
  SPIEL_CHECK_FALSE(state->IsTerminal());
}

void TieFirstTrickWinnerSecondTakesHandTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {33, 3, 20, 13, 35, 14});

  state->ApplyAction(33);
  state->ApplyAction(3);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);

  state->ApplyAction(20);
  state->ApplyAction(13);

  SPIEL_CHECK_FALSE(state->IsTerminal());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
}

void FirstWonSecondTiedWinnerFirstTakesHandTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 13, 33, 3, 35, 14});

  state->ApplyAction(20);
  state->ApplyAction(13);

  state->ApplyAction(33);
  state->ApplyAction(3);

  SPIEL_CHECK_FALSE(state->IsTerminal());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_GT(returns[0], 0);
}

void RetrucoDeclinePointsTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});

  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kRejectBetAction);

  SPIEL_CHECK_FALSE(state->IsTerminal());
  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_LT(returns[0], 0);
}

void EnvidoIllegalInSecondTrickTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});

  state->ApplyAction(0);
  state->ApplyAction(10);

  auto legal_actions = state->LegalActions();

  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kEnvidoAction) == legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRealEnvidoAction) == legal_actions.end());
  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kFaltaEnvidoAction) == legal_actions.end());

  SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                             kRaiseTrucoAction) != legal_actions.end());
}

void TensorShapeTest() {
  std::shared_ptr<const Game> game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<int> obs_shape = game->ObservationTensorShape();
  int expected_size = 0;
  expected_size += 2;
  expected_size += 40;
  expected_size += 240;
  expected_size += 15;
  expected_size += 2;
  expected_size += 4;
  expected_size += 12;
  expected_size += 2;
  expected_size += 2;
  expected_size += 2;

  int calculated_size = obs_shape[0];
  SPIEL_CHECK_EQ(calculated_size, expected_size);
}

void TrickAnalysisTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("truco");
  std::unique_ptr<State> state = game->NewInitialState();

  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }

  state->ApplyAction(state->LegalActions()[0]);
  state->ApplyAction(state->LegalActions()[0]);

  std::vector<float> tensor(game->ObservationTensorSize());
  state->ObservationTensor(0, absl::MakeSpan(tensor));

  int offset = 2 + 40 + 240;

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
  auto state = DealFixedHand(game, {33, 3, 34, 4, 13, 20});

  state->ApplyAction(33);
  state->ApplyAction(3);

  state->ApplyAction(34);
  state->ApplyAction(4);

  SPIEL_CHECK_FALSE(state->IsChanceNode());
  state->ApplyAction(13);
  state->ApplyAction(20);

  const auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_LT(returns[0], 0);
}

void ThirdTrickLeaderTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 13, 3, 10, 1, 26});

  state->ApplyAction(20);
  state->ApplyAction(13);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction(3);
  state->ApplyAction(26);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void EnvidoEnvidoRejectTest() {
  auto game = LoadGame("truco");
  auto state = DealFixedHand(game, {20, 26, 36, 25, 2, 24});

  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kEnvidoAction);
  state->ApplyAction(kRejectBetAction);

  auto returns = state->Returns();
  SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  SPIEL_CHECK_LT(returns[0], 0);
}

// Test that verifies the potential-based reward shaping properties
void PotentialBasedRewardShapingTest() {
  auto game = LoadGame("truco");

  // Test 1: Verify terminal returns are exactly +1/-1
  {
    auto state = DealFixedHand(game, {34, 0, 35, 1, 36, 2});
    state->ApplyAction(kFaltaEnvidoAction);
    state->ApplyAction(kAcceptBetAction);
    SPIEL_CHECK_TRUE(state->IsTerminal());
    auto returns = state->Returns();
    SPIEL_CHECK_EQ(returns[0], 1.0);
    SPIEL_CHECK_EQ(returns[1], -1.0);
  }

  // Test 2: Verify rewards sum to zero (zero-sum property)
  {
    auto state = DealFixedHand(game, {20, 26, 36, 25, 2, 24});
    state->ApplyAction(kEnvidoAction);
    state->ApplyAction(kAcceptBetAction);
    auto rewards = state->Rewards();
    SPIEL_CHECK_LT(std::abs(rewards[0] + rewards[1]), 1e-9);
  }

  // Test 3: Verify that returns accumulate correctly (shaped rewards telescope)
  // Play through multiple hands and verify returns at terminal
  {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(state->LegalActions()[0]);
      } else {
        state->ApplyAction(state->LegalActions()[0]);
      }
    }
    auto returns = state->Returns();
    SPIEL_CHECK_TRUE(std::abs(returns[0]) == 1.0);
    SPIEL_CHECK_EQ(returns[0] + returns[1], 0);
  }

  // Test 4: Verify potential formula directly
  // Starting position: both at 0 points, potential should be 0 for both
  {
    auto state = DealFixedHand(game, {0, 10, 1, 11, 2, 12});
    // Initial returns should be 0
    auto returns = state->Returns();
    SPIEL_CHECK_EQ(returns[0], 0);
    SPIEL_CHECK_EQ(returns[1], 0);

    // After P0 wins 1 point (truco decline):
    // New potential for P0: 1/30 = 0.0333...
    // New potential for P1: -1/30 = -0.0333...
    // Shaped reward = new_potential - old_potential = 0.0333... for P0
    state->ApplyAction(kRaiseTrucoAction);
    state->ApplyAction(kRejectBetAction);
    auto returns_after = state->Returns();
    double expected_delta = 1.0 / static_cast<double>(kTargetScore);
    SPIEL_CHECK_LT(std::abs(returns_after[0] - expected_delta), 1e-9);
    SPIEL_CHECK_LT(std::abs(returns_after[1] + expected_delta), 1e-9);
  }
}

// Test that shaped rewards properly track score changes
void ShapedRewardsTrackScoreTest() {
  auto game = LoadGame("truco");

  // Win 2 points with truco
  auto state = DealFixedHand(game, {20, 10, 21, 11, 22, 12});
  state->ApplyAction(kRaiseTrucoAction);
  state->ApplyAction(kAcceptBetAction);
  state->ApplyAction(20);
  state->ApplyAction(10);
  state->ApplyAction(21);
  state->ApplyAction(11);

  // P0 won 2 points
  // Expected shaped return = (2 - 0) / 30 = 2/30 ≈ 0.0667
  auto returns = state->Returns();
  double expected = 2.0 / static_cast<double>(kTargetScore);
  SPIEL_CHECK_LT(std::abs(returns[0] - expected), 1e-9);
  SPIEL_CHECK_LT(std::abs(returns[1] + expected), 1e-9);
}

// Test utility bounds
void UtilityBoundsTest() {
  auto game = LoadGame("truco");
  // Small epsilon for floating point tolerance
  SPIEL_CHECK_LT(std::abs(game->MinUtility() - (-1.0)), 1e-6);
  SPIEL_CHECK_LT(std::abs(game->MaxUtility() - 1.0), 1e-6);
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
  open_spiel::truco::EnvidoEnvidoRejectTest();

  // Potential-based reward shaping tests
  open_spiel::truco::PotentialBasedRewardShapingTest();
  open_spiel::truco::ShapedRewardsTrackScoreTest();
  open_spiel::truco::UtilityBoundsTest();
}
