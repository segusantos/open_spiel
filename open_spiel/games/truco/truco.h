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

// Truco Argentino is a popular trick-taking card game from South America.
// https://es.wikipedia.org/wiki/Truco_argentino
//
// This implementation follows the 2-player Argentine rules without Flor.
//
// Parameters:
//   "players"          int    number of players               (default = 2)
//   "starting_player"  int    which player starts             (default = 0)

#ifndef OPEN_SPIEL_GAMES_TRUCO_H_
#define OPEN_SPIEL_GAMES_TRUCO_H_

#include <array>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace truco {

inline constexpr int kDefaultPlayers = 2;
inline constexpr int kDefaultStartingPlayer = 0;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 10;
inline constexpr int kNumCards = kNumSuits * kNumRanks;
inline constexpr int kHandSize = 3;
inline constexpr int kNumTricks = 3;
inline constexpr Player kTiePlayer = -2;
inline constexpr int kNumEnvidoTypes = 3;
inline constexpr int kMaxEnvidoSequenceActions = 4;
inline constexpr int kEnvidoSequenceTensorSize =
    kMaxEnvidoSequenceActions * kNumEnvidoTypes;
inline constexpr int kTrucoLevelBits = 4;
inline constexpr int kEnvidoStateBits = 2;
inline constexpr int kTargetScore = 30;

inline constexpr int kEnvidoAction = kNumCards;
inline constexpr int kRealEnvidoAction = kNumCards + 1;
inline constexpr int kFaltaEnvidoAction = kNumCards + 2;
inline constexpr int kRaiseTrucoAction = kNumCards + 3;
inline constexpr int kAcceptBetAction = kNumCards + 4;
inline constexpr int kRejectBetAction = kNumCards + 5;
inline constexpr int kNumSpecialActions = 6;
inline constexpr int kNumDistinctTrucoActions = kNumCards + kNumSpecialActions;
inline constexpr int kMaxBettingActions = 10;

inline bool IsCardAction(Action action) {
  return action >= 0 && action < kNumCards;
}
inline bool IsEnvidoCallAction(Action action) {
  return action == kEnvidoAction || action == kRealEnvidoAction ||
         action == kFaltaEnvidoAction;
}
inline bool IsTrucoCallAction(Action action) {
  return action == kRaiseTrucoAction;
}
inline bool IsResponseAction(Action action) {
  return action == kAcceptBetAction || action == kRejectBetAction;
}

enum class PendingResponse { kNone, kEnvido, kTruco };

enum class EnvidoCall { kEnvido, kRealEnvido, kFaltaEnvido };

std::string PendingResponseToString(PendingResponse pending);
std::ostream& operator<<(std::ostream& stream, PendingResponse pending);

class TrucoGame;
class TrucoObserver;

enum class Suit { Basto, Copa, Espada, Oro };

enum class Rank {
  Ancho = 1,
  Dos = 2,
  Tres = 3,
  Cuatro = 4,
  Cinco = 5,
  Seis = 6,
  Siete = 7,
  Sota = 10,
  Caballo = 11,
  Rey = 12
};

struct Card {
  Card(Suit suit, Rank rank, int uid, int truco_value, int envido_value);

  const Suit suit_;
  const Rank rank_;

  const int uid_;
  const int truco_value_;
  const int envido_value_;

  std::string ToString() const;
};

std::vector<Card> CreateDeck();

struct Trick {
  Trick() = default;
  std::vector<Action> cards;
  std::vector<Player> players;
};

class TrucoState : public State {
 public:
  explicit TrucoState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  const std::vector<int>& PlayerHand(Player player) const;
  Player starting_player() const { return starting_player_; }
  int current_trick() const { return current_trick_index_; }
  const std::vector<Player>& trick_results() const { return trick_results_; }
  const std::vector<Trick>& trick_history() const { return tricks_; }
  Player winner() const { return winner_; }
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }

 protected:
  void DoApplyAction(Action move) override;

 private:
  friend class TrucoObserver;
  const TrucoGame& ParentGame() const;
  void DealCard(Action card);
  void ApplyPlayAction(Action card);
  void ApplyEnvidoCall(Action action);
  void ApplyTrucoCall(Action action);
  void HandleResponseAction(Action action);
  Player EvaluateTrick(const Trick& trick) const;
  void FinishTrick(Player trick_winner);
  void MaybeResolveHand(Player latest_trick_winner);
  std::vector<Action> LegalChanceOutcomes() const;
  Player NextPlayerInTrick(Player after) const;
  std::vector<Action> LegalEnvidoResponseActions() const;
  std::vector<Action> LegalTrucoResponseActions() const;
  std::vector<Action> LegalRegularActions() const;
  bool CanCallEnvido(Player player) const;
  bool EnvidoWindowOpen() const;
  bool CanRaiseTruco(Player player, Action action) const;
  void BeginPendingResponse(PendingResponse type, Player caller);
  void RestoreTurnAfterBet();
  void ResolveEnvidoAcceptance();
  void ResolveEnvidoDecline();
  void ResolveTrucoAcceptance();
  void ResolveTrucoDecline();
  void AwardPoints(Player player, int points);
  double Potential(Player player) const;
  void StartNewHand();
  int SumEnvidoPoints(bool include_last) const;
  int EnvidoCallValue(EnvidoCall call) const;
  int FaltaEnvidoValue(Player winner) const;
  int ComputeEnvidoScore(const std::vector<int>& cards) const;
  std::vector<std::vector<int>> FindCardCombinationsWithEnvidoScore(
      const std::vector<int>& available_cards, int num_cards,
      int target_score) const;
  std::vector<std::vector<int>> FindCardCombinationsWithEnvidoScoreGivenFixed(
      const std::vector<int>& available_cards, int num_cards,
      const std::vector<int>& fixed_cards, int target_score) const;
  int PlayerEnvidoScore(Player player) const;
  Player DetermineEnvidoWinner() const;
  Player Opponent(Player player) const { return 1 - player; }

  Player cur_player_;
  Player starting_player_;
  Player mano_;
  Player winner_;
  int cards_dealt_;
  int cards_played_in_trick_;
  int current_trick_index_;
  Player trick_leader_;
  std::vector<std::vector<int>> player_hands_;
  std::vector<Player> card_owner_;
  std::vector<bool> card_played_;
  std::vector<bool> played_current_trick_;
  std::vector<int> trick_wins_;
  std::vector<Player> trick_results_;
  std::vector<Trick> tricks_;
  std::vector<double> returns_;
  std::vector<double> rewards_;
  std::vector<int> game_points_;
  int num_hands_played_ = 0;
  bool terminal_ = false;

  // Betting/envido state.
  PendingResponse pending_response_ = PendingResponse::kNone;
  Player player_turn_before_bet_ = kInvalidPlayer;

  // Envido sequence data.
  std::vector<EnvidoCall> envido_sequence_;
  Player envido_last_caller_ = kInvalidPlayer;
  bool envido_resolved_ = false;
  bool envido_locked_ = false;
  std::array<int, 2> revealed_envido_scores_ = {-1, -1};

  // Truco betting data.
  int truco_level_ = 1;
  int pending_truco_target_ = 0;
  Player pending_truco_caller_ = kInvalidPlayer;
  Player truco_next_raiser_ = kInvalidPlayer;

  struct TrucoResponseState {
    PendingResponse pending_response;
    Player pending_truco_caller;
    int pending_truco_target;
    Player cur_player;
  };
  std::vector<TrucoResponseState> response_stack_;
};

class TrucoGame : public Game {
 public:
  explicit TrucoGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctTrucoActions; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    int actions_per_hand = kHandSize * num_players_ + kHandSize * num_players_ +
                           kMaxBettingActions;
    return 2 * kTargetScore * actions_per_hand;
  }
  int MaxChanceNodesInHistory() const override {
    return 2 * kTargetScore * num_players_ * kHandSize;
  }

  const std::vector<Card>& deck() const { return deck_; }
  const Card& CardById(int card) const { return deck_.at(card); }
  Player starting_player() const { return starting_player_; }
  Player mano() const { return starting_player_; }

  std::string ActionToString(Player player, Action action) const override;
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  std::shared_ptr<Observer> default_observer_;
  std::shared_ptr<Observer> info_state_observer_;

 private:
  int num_players_;
  int total_cards_;
  int starting_player_;
  const std::vector<Card> deck_;
};

}  // namespace truco
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_TRUCO_H_
