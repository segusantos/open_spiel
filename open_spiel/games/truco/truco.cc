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
#include <array>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace truco {

std::string PendingResponseToString(PendingResponse pending) {
  switch (pending) {
    case PendingResponse::kNone:
      return "No response pending";
    case PendingResponse::kEnvido:
      return "Envido response pending";
    case PendingResponse::kTruco:
      return "Truco response pending";
  }
  SpielFatalError("Unknown PendingResponse value");
}

std::ostream& operator<<(std::ostream& stream, PendingResponse pending) {
  stream << PendingResponseToString(pending);
  return stream;
}

namespace {

const GameType kGameType{
    /*short_name=*/"truco",
    /*long_name=*/"Truco",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kRewards,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(kDefaultPlayers)},
     {"starting_player", GameParameter(kDefaultStartingPlayer)}}};  // NOLINT

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new TrucoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string CardString(const TrucoGame& game, Action action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumCards);
  return game.CardById(action).ToString();
}

std::string EnvidoCallToString(EnvidoCall call) {
  switch (call) {
    case EnvidoCall::kEnvido:
      return "Envido called";
    case EnvidoCall::kRealEnvido:
      return "Real Envido called";
    case EnvidoCall::kFaltaEnvido:
      return "Falta Envido called";
  }
  SpielFatalError("Unknown EnvidoCall value");
}

std::string SpecialActionToString(Action action) {
  switch (action) {
    case kEnvidoAction:
      return "Envido";
    case kRealEnvidoAction:
      return "Real Envido";
    case kFaltaEnvidoAction:
      return "Falta Envido";
    case kRaiseTrucoAction:
      return "Raise Truco";
    case kAcceptBetAction:
      return "Quiero";
    case kRejectBetAction:
      return "No Quiero";
    case kNewHandAction:
      return "New Hand";
  }
  SpielFatalError("Unknown SpecialAction value");
}

}  // namespace

// The Observer class is responsible for creating representations of the game
// state for use in learning algorithms. It handles both string and tensor
// representations, and any combination of public information and private
// information (none, observing player only, or all players).

class TrucoObserver : public Observer {
 public:
  explicit TrucoObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  //
  // These helper methods each write a piece of the tensor observation.
  //

  // Identity of the observing player. One-hot vector of size num_players.
  static void WriteObservingPlayer(const TrucoState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  // Private cards of the observing player. One-hot vector of size kNumCards.
  static void WritePrivateCards(const TrucoState& state, int player,
                                Allocator* allocator) {
    auto out = allocator->Get("hand", {kNumCards});
    for (int card : state.PlayerHand(player)) {
      out.at(card) = 1;
    }
  }

  // Private cards of all players. Tensor of shape [num_players, kNumCards].
  static void WriteAllPrivateCards(const TrucoState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("hands", {state.num_players_, kNumCards});
    for (Player p = 0; p < state.num_players_; ++p) {
      for (int card : state.PlayerHand(p)) {
        out.at(p, card) = 1;
      }
    }
  }

  // History of all tricks played. Shape [kNumTricks, num_players, kNumCards].
  // History of all tricks played. Shape [kNumTricks, num_players, kNumCards].
  static void WriteTrickHistory(const TrucoState& state, Allocator* allocator) {
    auto out =
        allocator->Get("tricks", {kNumTricks, state.num_players_, kNumCards});
    for (int trick = 0; trick < kNumTricks; ++trick) {
      const Trick& t = state.trick_history()[trick];
      for (int idx = 0; idx < t.cards.size(); ++idx) {
        out.at(trick, idx, t.cards[idx]) = 1;
      }
    }
  }

  // Envido betting sequence. Vector of size kEnvidoTensorSize.
  static void WriteEnvidoState(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("envido", {kEnvidoTensorSize});
    for (const EnvidoCall& call : state.envido_sequence_) {
      out.at(static_cast<int>(call)) = 1;
    }
    if (state.envido_sequence_.empty() && state.envido_resolved_) {
      out.at(0) = 1;
    }
  }

  // Truco betting level (1-4). Vector of size kTrucoTensorSize.
  static void WriteTrucoState(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("truco", {kTrucoTensorSize});
    int accepted_level = state.truco_level_;
    if (accepted_level >= 1 && accepted_level <= kTrucoTensorSize) {
      out.at(accepted_level - 1) = 1;
    }
    if (state.pending_response_ == PendingResponse::kTruco &&
        state.pending_truco_target_ > 0) {
      int pending_level = state.pending_truco_target_;
      if (pending_level >= 1 && pending_level <= kTrucoTensorSize) {
        out.at(pending_level - 1) = 1;
      }
    }
  }

  // Pending bets and resolution status. Vector of size kPendingTensorSize.
  static void WritePendingState(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("pending", {kPendingTensorSize});
    if (state.pending_response_ == PendingResponse::kEnvido) {
      out.at(0) = 1;
    } else if (state.pending_response_ == PendingResponse::kTruco) {
      out.at(1) = 1;
    }
    if (state.envido_resolved_) {
      out.at(2) = 1;
    }
  }

  // Game scores normalized to [0, 1]. Vector of size num_players.
  // This is critical for agents to learn optimal Falta Envido strategies,
  // which depend on whether opponents are in "malas" (0-14) or "buenas"
  // (15-29).
  static void WriteGameScores(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("game_scores", {state.num_players_});
    for (Player p = 0; p < state.num_players_; ++p) {
      // Normalize scores to [0, 1] range (game plays to 30)
      out.at(p) = static_cast<float>(state.game_points_[p]) / kTargetScore;
    }
  }

  // Mano (starting player) indicator. Vector of size num_players.
  // Critical for: tie-breaking (mano wins ties), Envido ties, strategic value.
  static void WriteMano(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("mano", {state.num_players_});
    out.at(state.mano_) = 1;
  }

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const auto& state =
        open_spiel::down_cast<const TrucoState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player.
    WriteObservingPlayer(state, player, allocator);

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WritePrivateCards(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPrivateCards(state, allocator);
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      WriteTrickHistory(state, allocator);
      WriteEnvidoState(state, allocator);
      WriteTrucoState(state, allocator);
      WritePendingState(state, allocator);
      WriteGameScores(state, allocator);
      WriteMano(state, allocator);
    }
  }

  // Writes an observation in string form. This provides a somewhat
  // human-readable representation of the game state from a player's
  // perspective.
  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const auto& state =
        open_spiel::down_cast<const TrucoState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    std::string result = absl::StrCat("[Observer:", player, "]");

    // Private information.
    if (iig_obs_type_.private_info != PrivateInfoType::kNone) {
      std::vector<std::string> cards;
      const TrucoGame& game = state.ParentGame();
      for (int card : state.PlayerHand(player)) {
        cards.push_back(CardString(game, card));
      }
      absl::StrAppend(&result, "[Hand:", absl::StrJoin(cards, ","), "]");
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "[Trick:", state.current_trick(), "]");
      if (!state.envido_sequence_.empty() || state.envido_resolved_) {
        std::vector<std::string> seq;
        for (const EnvidoCall& call : state.envido_sequence_) {
          seq.push_back(EnvidoCallToString(call));
        }
        absl::StrAppend(&result, "[Envido:", absl::StrJoin(seq, ">"), "]");
      }
      absl::StrAppend(&result, "[TrucoLvl:", state.truco_level_, "]");
      // Game scores are critical for Falta Envido decisions (malas vs buenas)
      absl::StrAppend(&result, "[GameScore:", state.game_points_[0], ",",
                      state.game_points_[1], "]");
    }

    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

// ---------------------------------------------------------------------------
// Card implementation
// ---------------------------------------------------------------------------

Card::Card(Suit suit, Rank rank, int uid, int truco_value, int envido_value)
    : suit_(suit),
      rank_(rank),
      uid_(uid),
      truco_value_(truco_value),
      envido_value_(envido_value) {}

std::string Card::ToString() const {
  std::string s = std::to_string(static_cast<int>(rank_));
  switch (suit_) {
    case Suit::Basto:
      absl::StrAppend(&s, " de Basto");
      break;
    case Suit::Copa:
      absl::StrAppend(&s, " de Copa");
      break;
    case Suit::Espada:
      absl::StrAppend(&s, " de Espada");
      break;
    case Suit::Oro:
      absl::StrAppend(&s, " de Oro");
      break;
  }
  return s;
}

constexpr int getTrucoValue(Suit suit, Rank rank) {
  switch (rank) {
    case Rank::Ancho:
      if (suit == Suit::Espada) return 14;
      if (suit == Suit::Basto) return 13;
      return 8;
    case Rank::Siete:
      if (suit == Suit::Espada) return 12;
      if (suit == Suit::Oro) return 11;
      return 4;
    case Rank::Tres:
      return 10;
    case Rank::Dos:
      return 9;
    case Rank::Rey:
      return 7;
    case Rank::Caballo:
      return 6;
    case Rank::Sota:
      return 5;
    case Rank::Seis:
      return 3;
    case Rank::Cinco:
      return 2;
    case Rank::Cuatro:
      return 1;
  }
  SpielFatalError("Unknown Suit/Rank combination");
}

constexpr int getEnvidoValue(Rank rank) {
  switch (rank) {
    case Rank::Ancho:
    case Rank::Dos:
    case Rank::Tres:
    case Rank::Cuatro:
    case Rank::Cinco:
    case Rank::Seis:
    case Rank::Siete:
      return static_cast<int>(rank);
    case Rank::Sota:
    case Rank::Caballo:
    case Rank::Rey:
      return 0;
  }
  SpielFatalError("Unknown Rank value");
}

std::vector<Card> CreateDeck() {
  std::vector<Card> deck;
  deck.reserve(kNumCards);
  int uid = 0;
  for (Suit suit : {Suit::Basto, Suit::Copa, Suit::Espada, Suit::Oro}) {
    for (Rank rank :
         {Rank::Ancho, Rank::Dos, Rank::Tres, Rank::Cuatro, Rank::Cinco,
          Rank::Seis, Rank::Siete, Rank::Sota, Rank::Caballo, Rank::Rey}) {
      deck.emplace_back(suit, rank, uid, getTrucoValue(suit, rank),
                        getEnvidoValue(rank));
      ++uid;
    }
  }
  return deck;
}

// ---------------------------------------------------------------------------
// TrucoState implementation
// ---------------------------------------------------------------------------

const TrucoGame& TrucoState::ParentGame() const {
  return open_spiel::down_cast<const TrucoGame&>(*game_);
}

TrucoState::TrucoState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),  // Start in chance phase for dealing
      starting_player_(ParentGame().starting_player()),
      mano_(starting_player_),
      winner_(kInvalidPlayer),
      cards_dealt_(0),
      cards_played_in_trick_(0),
      current_trick_index_(0),
      trick_leader_(starting_player_),
      player_hands_(num_players_),
      card_owner_(kNumCards, kInvalidPlayer),
      card_played_(kNumCards, false),
      played_current_trick_(num_players_, false),
      trick_wins_(num_players_, 0),
      trick_results_(kNumTricks, kInvalidPlayer),
      tricks_(kNumTricks),
      returns_(num_players_, 0.0),
      rewards_(num_players_, 0.0),
      game_points_(num_players_, 0) {}

Player TrucoState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  return cur_player_;
}

std::string TrucoState::ActionToString(Player player, Action move) const {
  return ParentGame().ActionToString(player, move);
}

std::string TrucoState::ToString() const {
  const TrucoGame& game = ParentGame();
  std::string result;

  // Header
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");
  absl::StrAppend(&result, "| Truco - Hand ", num_hands_played_ + 1, "\n");
  absl::StrAppend(&result, "| Score: P0 [ ", game_points_[0], " ]  vs  P1 [ ",
                  game_points_[1], " ]\n");
  absl::StrAppend(&result, "| Mano: P", mano_, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  // P1 Hand (Top)
  absl::StrAppend(&result, "| P1 Hand: ");
  for (int card : player_hands_[1]) {
    if (!card_played_[card]) {
      absl::StrAppend(&result, "[", CardString(game, card), "] ");
    }
  }
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  // Tricks (Middle)
  absl::StrAppend(&result, "| Tricks:\n");
  for (int t = 0; t < kNumTricks; ++t) {
    const Trick& trick = tricks_[t];
    if (trick.cards.empty()) {
      absl::StrAppend(&result, "| ", t + 1, ". (not started)\n");
    } else {
      absl::StrAppend(&result, "| ", t + 1, ". ");
      for (int i = 0; i < trick.cards.size(); ++i) {
        absl::StrAppend(&result, "P", trick.players[i], ": ",
                        CardString(game, trick.cards[i]));
        if (i < trick.cards.size() - 1) absl::StrAppend(&result, " | ");
      }
      if (trick_results_[t] != kInvalidPlayer) {
        absl::StrAppend(&result, " -> Winner: P", trick_results_[t]);
      }
      absl::StrAppend(&result, "\n");
    }
  }
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  // P0 Hand (Bottom)
  absl::StrAppend(&result, "| P0 Hand: ");
  for (int card : player_hands_[0]) {
    if (!card_played_[card]) {
      absl::StrAppend(&result, "[", CardString(game, card), "] ");
    }
  }
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  // Betting Info
  absl::StrAppend(&result, "| Betting:\n");
  if (!envido_sequence_.empty() || envido_resolved_) {
    absl::StrAppend(&result, "|   Envido: ");
    for (size_t i = 0; i < envido_sequence_.size(); ++i) {
      absl::StrAppend(&result, EnvidoCallToString(envido_sequence_[i]));
      if (i < envido_sequence_.size() - 1) absl::StrAppend(&result, " > ");
    }
    if (pending_response_ == PendingResponse::kEnvido) {
      absl::StrAppend(&result, " (Pending Response)");
    } else if (envido_resolved_) {
      absl::StrAppend(&result, " (Resolved)");
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "|   Truco Level: ", truco_level_);
  if (pending_response_ == PendingResponse::kTruco) {
    absl::StrAppend(&result, " (Pending Response)");
  }
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  // Status
  if (IsTerminal()) {
    absl::StrAppend(&result, "| GAME OVER\n");
    absl::StrAppend(&result, "| Final Returns: P0=", returns_[0],
                    "  P1=", returns_[1], "\n");
  } else {
    absl::StrAppend(
        &result, "| Next Action: ",
        (cur_player_ == kChancePlayerId ? "Chance"
                                       : absl::StrCat("P", cur_player_)),
        "\n");
  }
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");
    
  return result;
}

bool TrucoState::IsTerminal() const { return terminal_; }

std::vector<double> TrucoState::Returns() const {
  // Returns track cumulative score across all hands
  return returns_;
}

std::vector<double> TrucoState::Rewards() const { return rewards_; }

std::string TrucoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  std::string info =
      absl::StrCat("P", player, "|M", mano_, "|T", current_trick_index_);
  std::vector<std::string> cards;
  const TrucoGame& game = ParentGame();
  for (int card : PlayerHand(player)) {
    cards.push_back(CardString(game, card));
  }
  absl::StrAppend(&info, "|H:", absl::StrJoin(cards, ","));
  int max_trick = std::min(current_trick_index_, kNumTricks - 1);
  for (int trick = 0; trick <= max_trick; ++trick) {
    const Trick& t = tricks_[trick];
    std::vector<std::string> trick_cards;
    for (int idx = 0; idx < t.cards.size(); ++idx) {
      trick_cards.push_back(absl::StrCat("P", t.players[idx], ":",
                                         CardString(game, t.cards[idx])));
    }
    absl::StrAppend(&info, "|R", trick, ":", absl::StrJoin(trick_cards, ","));
  }
  if (!envido_sequence_.empty() || envido_resolved_) {
    std::vector<std::string> calls;
    for (const EnvidoCall& call : envido_sequence_) {
      calls.push_back(EnvidoCallToString(call));
    }
    absl::StrAppend(&info, "|E:", absl::StrJoin(calls, ">"));
    if (pending_response_ == PendingResponse::kEnvido) {
      absl::StrAppend(&info, "(P)");
    } else if (envido_resolved_) {
      absl::StrAppend(&info, "(R)");
    }
  }
  absl::StrAppend(&info, "|TL:", truco_level_);
  if (pending_response_ == PendingResponse::kTruco) {
    absl::StrAppend(&info, "|TP:", pending_truco_target_, "C",
                    pending_truco_caller_);
  }
  return info;
}

std::string TrucoState::ObservationString(Player player) const {
  const TrucoGame& game = ParentGame();
  return game.default_observer_->StringFrom(*this, player);
}

void TrucoState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const TrucoGame& game = ParentGame();
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void TrucoState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const TrucoGame& game = ParentGame();
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> TrucoState::Clone() const {
  return absl::make_unique<TrucoState>(*this);
}

std::vector<std::pair<Action, double>> TrucoState::ChanceOutcomes() const {
  SPIEL_CHECK_EQ(cur_player_, kChancePlayerId);
  std::vector<std::pair<Action, double>> outcomes;
  const std::vector<Action> legal = LegalChanceOutcomes();
  if (legal.empty()) return outcomes;
  double prob = 1.0 / static_cast<double>(legal.size());
  for (Action action : legal) {
    outcomes.push_back({action, prob});
  }
  return outcomes;
}

std::vector<Action> TrucoState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (hand_over_) {
    return {kNewHandAction};
  }
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }
  if (pending_response_ == PendingResponse::kEnvido) {
    return LegalEnvidoResponseActions();
  }
  if (pending_response_ == PendingResponse::kTruco) {
    return LegalTrucoResponseActions();
  }
  std::vector<Action> actions = LegalRegularActions();
  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> TrucoState::LegalRegularActions() const {
  std::vector<Action> actions;
  for (Action card = 0; card < kNumCards; ++card) {
    if (card_owner_[card] == cur_player_ && !card_played_[card]) {
      actions.push_back(card);
    }
  }
  if (CanCallEnvido(cur_player_)) {
    actions.push_back(kEnvidoAction);
    actions.push_back(kRealEnvidoAction);
    actions.push_back(kFaltaEnvidoAction);
  }
  if (CanRaiseTruco(cur_player_, kRaiseTrucoAction)) {
    actions.push_back(kRaiseTrucoAction);
  }
  return actions;
}

std::vector<Action> TrucoState::LegalEnvidoResponseActions() const {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
  std::vector<Action> actions = {kAcceptBetAction, kRejectBetAction};
  if (!envido_sequence_.empty()) {
    const EnvidoCall last_call = envido_sequence_.back();
    bool has_real =
        std::find(envido_sequence_.begin(), envido_sequence_.end(),
                  EnvidoCall::kRealEnvido) != envido_sequence_.end();
    if (last_call == EnvidoCall::kEnvido && !has_real) {
      actions.push_back(kRealEnvidoAction);
    }
    if (last_call != EnvidoCall::kFaltaEnvido) {
      actions.push_back(kFaltaEnvidoAction);
    }
  }
  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> TrucoState::LegalTrucoResponseActions() const {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kTruco);
  std::vector<Action> actions = {kAcceptBetAction, kRejectBetAction};
  if (pending_truco_target_ < 4) {
    actions.push_back(kRaiseTrucoAction);
  }

  // "El envido está primero": Can call Envido in response to Truco if window is
  // open
  if (EnvidoWindowOpen() && !envido_resolved_ && !envido_locked_) {
    actions.push_back(kEnvidoAction);
    actions.push_back(kRealEnvidoAction);
    actions.push_back(kFaltaEnvidoAction);
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

bool TrucoState::CanCallEnvido(Player player) const {
  if (player != cur_player_) return false;
  if (pending_response_ != PendingResponse::kNone) return false;
  if (envido_resolved_ || envido_locked_) return false;
  if (!EnvidoWindowOpen()) return false;
  return true;
}

bool TrucoState::EnvidoWindowOpen() const { return current_trick_index_ == 0; }

bool TrucoState::CanRaiseTruco(Player player, Action action) const {
  if (player != cur_player_) return false;
  if (pending_response_ != PendingResponse::kNone) return false;
  if (action != kRaiseTrucoAction) return false;

  if (truco_level_ == 1) return true;
  if (truco_level_ < 4 && truco_next_raiser_ == player) return true;
  return false;
}

std::vector<int> TrucoState::PlayerHand(Player player) const {
  return player_hands_[player];
}

std::unique_ptr<State> TrucoState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = Clone();
  auto* truco_clone = open_spiel::down_cast<TrucoState*>(clone.get());

  Player opponent = 1 - player_id;
  std::vector<int> candidates;
  std::vector<int> opponent_slots;

  // Identify unplayed cards in opponent's hand
  for (size_t i = 0; i < truco_clone->player_hands_[opponent].size(); ++i) {
    int card = truco_clone->player_hands_[opponent][i];
    if (!truco_clone->card_played_[card]) {
      opponent_slots.push_back(i);
      candidates.push_back(card);
    }
  }

  // Identify cards in the deck
  for (int card = 0; card < kNumCards; ++card) {
    if (truco_clone->card_owner_[card] == kInvalidPlayer) {
      candidates.push_back(card);
    }
  }

  // Shuffle candidates
  // Use the provided rng to seed a generator
  // Note: rng() returns double in [0, 1).
  uint32_t seed =
      static_cast<uint32_t>(rng() * std::numeric_limits<uint32_t>::max());
  std::mt19937 gen(seed);
  std::shuffle(candidates.begin(), candidates.end(), gen);

  // Redistribute to opponent
  for (int slot : opponent_slots) {
    int new_card = candidates.back();
    candidates.pop_back();
    truco_clone->player_hands_[opponent][slot] = new_card;
    truco_clone->card_owner_[new_card] = opponent;
  }

  // Redistribute to deck
  for (int card : candidates) {
    truco_clone->card_owner_[card] = kInvalidPlayer;
  }

  return clone;
}

void TrucoState::DoApplyAction(Action move) {
  // Reset rewards for all actions.
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  if (hand_over_) {
    SPIEL_CHECK_EQ(move, kNewHandAction);
    StartNewHand();
    return;
  }

  if (cur_player_ == kChancePlayerId) {
    // During dealing phase
    DealCard(move);
  } else {
    // During play phase
    if (IsCardAction(move)) {
      SPIEL_CHECK_EQ(pending_response_, PendingResponse::kNone);
      ApplyPlayAction(move);
    } else if (IsEnvidoCallAction(move)) {
      ApplyEnvidoCall(move);
    } else if (IsTrucoCallAction(move)) {
      ApplyTrucoCall(move);
    } else if (IsResponseAction(move)) {
      HandleResponseAction(move);
    } else {
      SpielFatalError("Unknown action");
    }
  }
}

void TrucoState::DealCard(Action card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  SPIEL_CHECK_EQ(card_owner_[card], kInvalidPlayer);

  Player recipient = cards_dealt_ % num_players_;
  player_hands_[recipient].push_back(card);
  card_owner_[card] = recipient;
  ++cards_dealt_;
  if (cards_dealt_ == num_players_ * kHandSize) {
    cur_player_ = starting_player_;
    trick_leader_ = starting_player_;
  }
}

void TrucoState::ApplyPlayAction(Action card) {
  SPIEL_CHECK_GE(card, 0);
  SPIEL_CHECK_LT(card, kNumCards);
  SPIEL_CHECK_FALSE(card_played_[card]);
  SPIEL_CHECK_EQ(card_owner_[card], cur_player_);
  SPIEL_CHECK_FALSE(played_current_trick_[cur_player_]);

  card_played_[card] = true;
  played_current_trick_[cur_player_] = true;

  Trick& trick = tricks_[current_trick_index_];
  trick.cards.push_back(card);
  trick.players.push_back(cur_player_);
  ++cards_played_in_trick_;

  if (cards_played_in_trick_ == num_players_) {
    Player trick_winner = EvaluateTrick(trick);
    FinishTrick(trick_winner);
  } else {
    cur_player_ = NextPlayerInTrick(cur_player_);
  }
}

void TrucoState::ApplyEnvidoCall(Action action) {
  // Calling envido doesn't immediately award points
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  EnvidoCall call;
  if (action == kEnvidoAction) {
    call = EnvidoCall::kEnvido;
  } else if (action == kRealEnvidoAction) {
    call = EnvidoCall::kRealEnvido;
  } else {
    call = EnvidoCall::kFaltaEnvido;
  }

  if (pending_response_ == PendingResponse::kTruco) {
    // "El envido está primero": Suspend Truco, start Envido
    response_stack_.push_back({pending_response_, pending_truco_caller_,
                               pending_truco_target_, cur_player_});
    pending_response_ = PendingResponse::kEnvido;
    // The player who called Envido (current player) becomes the last caller of
    // Envido The opponent must respond to Envido
    envido_last_caller_ = cur_player_;
    cur_player_ = Opponent(cur_player_);
  } else if (pending_response_ == PendingResponse::kNone) {
    Player caller = cur_player_;
    BeginPendingResponse(PendingResponse::kEnvido, caller);
    envido_last_caller_ = caller;
  } else {
    SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
    // Raising over an existing envido implicitly accepts the previous wager.
    // The raiser is the current player, so save them before flipping.
    envido_last_caller_ = cur_player_;
    cur_player_ = Opponent(cur_player_);
  }
  envido_sequence_.push_back(call);
}

void TrucoState::ApplyTrucoCall(Action action) {
  // Calling truco doesn't immediately award points
  std::fill(rewards_.begin(), rewards_.end(), 0.0);
  SPIEL_CHECK_EQ(action, kRaiseTrucoAction);

  int target = 0;
  if (pending_response_ == PendingResponse::kTruco) {
    target = pending_truco_target_ + 1;
  } else {
    target = truco_level_ + 1;
  }
  SPIEL_CHECK_LE(target, 4);

  if (pending_response_ == PendingResponse::kTruco) {
    SPIEL_CHECK_GT(pending_truco_target_, 0);
    // Accept the previous raise and launch the next one.
    truco_level_ = pending_truco_target_;
    pending_truco_target_ = target;
    pending_truco_caller_ = cur_player_;
    cur_player_ = Opponent(cur_player_);
    return;
  }

  Player caller = cur_player_;
  BeginPendingResponse(PendingResponse::kTruco, caller);
  pending_truco_caller_ = caller;
  pending_truco_target_ = target;
}

void TrucoState::HandleResponseAction(Action action) {
  SPIEL_CHECK_NE(pending_response_, PendingResponse::kNone);
  if (pending_response_ == PendingResponse::kEnvido) {
    if (action == kAcceptBetAction) {
      ResolveEnvidoAcceptance();
    } else {
      ResolveEnvidoDecline();
    }
  } else {
    if (action == kAcceptBetAction) {
      ResolveTrucoAcceptance();
    } else {
      ResolveTrucoDecline();
    }
  }
}

void TrucoState::BeginPendingResponse(PendingResponse type, Player caller) {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kNone);
  pending_response_ = type;
  if (player_turn_before_bet_ == kInvalidPlayer) {
    // Save whose turn it is to play a card. After the bet resolves, this player
    // will continue playing. The caller is the player whose turn it was when
    // the bet started (they chose to bet instead of playing a card).
    player_turn_before_bet_ = caller;
  }
  // The opponent must respond to the bet.
  cur_player_ = Opponent(caller);
}

void TrucoState::RestoreTurnAfterBet() {
  SPIEL_CHECK_NE(player_turn_before_bet_, kInvalidPlayer);
  cur_player_ = player_turn_before_bet_;
  player_turn_before_bet_ = kInvalidPlayer;
}

void TrucoState::ResolveEnvidoAcceptance() {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
  Player winner = DetermineEnvidoWinner();
  int points = 0;
  if (!envido_sequence_.empty() &&
      envido_sequence_.back() == EnvidoCall::kFaltaEnvido) {
    // Falta Envido is not cumulative: only its value is awarded
    points = FaltaEnvidoValue();
  } else {
    points = SumEnvidoPoints(/*include_last=*/true);
  }
  AwardPoints(winner, points);

  envido_resolved_ = true;
  envido_locked_ = true;
  envido_sequence_.clear();
  envido_last_caller_ = kInvalidPlayer;

  if (!response_stack_.empty()) {
    // Restore Truco state
    TrucoResponseState state = response_stack_.back();
    response_stack_.pop_back();
    pending_response_ = state.pending_response;
    pending_truco_caller_ = state.pending_truco_caller;
    pending_truco_target_ = state.pending_truco_target;
    cur_player_ = state.cur_player;
  } else {
    pending_response_ = PendingResponse::kNone;
    RestoreTurnAfterBet();
  }
}

void TrucoState::ResolveEnvidoDecline() {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
  int accepted_points = SumEnvidoPoints(/*include_last=*/false);
  int decline_points = accepted_points + 1;
  SPIEL_CHECK_NE(envido_last_caller_, kInvalidPlayer);
  Player winner = envido_last_caller_;
  AwardPoints(winner, decline_points);

  envido_resolved_ = true;
  envido_locked_ = true;
  envido_sequence_.clear();
  envido_last_caller_ = kInvalidPlayer;

  if (!response_stack_.empty()) {
    // Restore Truco state
    TrucoResponseState state = response_stack_.back();
    response_stack_.pop_back();
    pending_response_ = state.pending_response;
    pending_truco_caller_ = state.pending_truco_caller;
    pending_truco_target_ = state.pending_truco_target;
    cur_player_ = state.cur_player;
  } else {
    pending_response_ = PendingResponse::kNone;
    RestoreTurnAfterBet();
  }
}

void TrucoState::ResolveTrucoAcceptance() {
  // Accepting truco doesn't immediately award points
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kTruco);
  SPIEL_CHECK_GT(pending_truco_target_, 0);
  truco_level_ = pending_truco_target_;
  pending_truco_target_ = 0;
  pending_truco_caller_ = kInvalidPlayer;
  pending_response_ = PendingResponse::kNone;

  // Accepting Truco closes the Envido window
  envido_locked_ = true;

  if (truco_level_ < 4) {
    truco_next_raiser_ = cur_player_;
  } else {
    truco_next_raiser_ = kInvalidPlayer;
  }
  RestoreTurnAfterBet();
}

void TrucoState::ResolveTrucoDecline() {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kTruco);
  SPIEL_CHECK_GT(pending_truco_target_, 1);
  int points = pending_truco_target_ - 1;
  if (points <= 0) points = 1;
  Player hand_winner = pending_truco_caller_;
  AwardPoints(hand_winner, points);
  pending_response_ = PendingResponse::kNone;
  pending_truco_target_ = 0;
  pending_truco_caller_ = kInvalidPlayer;

  // Truco decline ends the hand immediately
  winner_ = hand_winner;
  num_hands_played_++;

  // Check if game reached 30 points or should continue to next hand
  // Since scores are now absolute positive integers, we just check >=
  // kTargetScore
  if (game_points_[0] >= kTargetScore || game_points_[1] >= kTargetScore) {
    terminal_ = true;
    cur_player_ = kTerminalPlayerId;
  } else {
    // Continue to next hand
    hand_over_ = true;
    cur_player_ = winner_;
  }
}

void TrucoState::AwardPoints(Player player, int points) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_GE(points, 0);
  if (points == 0) return;

  // 1. Update absolute game scores (The "Scoreboard")
  // Truco is a race: points only go up (or stay same), they don't subtract from
  // opponent.
  int old_score = game_points_[player];
  int new_score = std::min(old_score + points, kTargetScore);
  int actual_points_gained = new_score - old_score;

  game_points_[player] = new_score;
  // Opponent's score does NOT change.

  // 2. Calculate Rewards (Zero-Sum for RL)
  // We reward the agent for the points gained. To keep the game zero-sum for RL
  // algorithms, we penalize the opponent by the same amount.
  std::fill(rewards_.begin(), rewards_.end(), 0.0);
  rewards_[player] = static_cast<double>(actual_points_gained);
  rewards_[Opponent(player)] = -static_cast<double>(actual_points_gained);

  // 3. Update Returns (Cumulative Zero-Sum Utility)
  returns_[player] += rewards_[player];
  returns_[Opponent(player)] += rewards_[Opponent(player)];
}

int TrucoState::SumEnvidoPoints(bool include_last) const {
  if (envido_sequence_.empty()) return 0;
  int limit = include_last ? envido_sequence_.size()
                           : static_cast<int>(envido_sequence_.size()) - 1;
  limit = std::max(limit, 0);
  int total = 0;
  for (int i = 0; i < limit; ++i) {
    total += EnvidoCallValue(envido_sequence_[i]);
  }
  return total;
}

int TrucoState::EnvidoCallValue(EnvidoCall call) const {
  switch (call) {
    case EnvidoCall::kEnvido:
      return 2;
    case EnvidoCall::kRealEnvido:
      return 3;
    case EnvidoCall::kFaltaEnvido:
      return FaltaEnvidoValue();
  }
  return 0;
}

int TrucoState::FaltaEnvidoValue() const {
  // Falta Envido: Winner gets points loser needs to reach 15 (if loser in
  // malas) or points loser needs to reach 30 (if loser in buenas) The "loser"
  // is the opponent of the current envido_last_caller_
  Player opponent = Opponent(envido_last_caller_);
  int opponent_score = game_points_[opponent];

  if (opponent_score < kMalasBuenasThreshold) {
    // Opponent in "malas" (0-14): winner gets points opponent needs to reach 15
    return kMalasBuenasThreshold - opponent_score;
  } else {
    // Opponent in "buenas" (15-29): winner gets points opponent needs to reach
    // 30
    return kTargetScore - opponent_score;
  }
}

int TrucoState::PlayerEnvidoScore(Player player) const {
  const TrucoGame& game = ParentGame();
  std::array<std::vector<int>, kNumSuits> suit_values;
  for (int card : player_hands_[player]) {
    const Card& c = game.CardById(card);
    suit_values[static_cast<int>(c.suit_)].push_back(c.envido_value_);
  }
  int best = 0;
  for (auto& vec : suit_values) {
    if (vec.size() >= 2) {
      std::sort(vec.begin(), vec.end(), std::greater<int>());
      best = std::max(best, 20 + vec[0] + vec[1]);
    }
  }
  if (best == 0) {
    for (auto& vec : suit_values) {
      if (!vec.empty()) {
        best = std::max(best, vec[0]);
      }
    }
  }
  return best;
}

Player TrucoState::DetermineEnvidoWinner() const {
  int score0 = PlayerEnvidoScore(0);
  int score1 = PlayerEnvidoScore(1);
  if (score0 > score1) {
    return 0;
  } else if (score1 > score0) {
    return 1;
  }
  return mano_;
}

Player TrucoState::EvaluateTrick(const Trick& trick) const {
  const TrucoGame& game = ParentGame();
  int best_value = -1;
  Player winner = kTiePlayer;
  for (int idx = 0; idx < trick.cards.size(); ++idx) {
    const Card& card = game.CardById(trick.cards[idx]);
    int value = card.truco_value_;
    if (value > best_value) {
      best_value = value;
      winner = trick.players[idx];
    } else if (value == best_value) {
      winner = kTiePlayer;
    }
  }
  return winner;
}

void TrucoState::FinishTrick(Player trick_winner) {
  trick_results_[current_trick_index_] = trick_winner;
  if (trick_winner != kTiePlayer) {
    ++trick_wins_[trick_winner];
  }

  // Finishing a trick might award points (if hand ends), or might not (if
  // continuing) MaybeResolveHand will call AwardPoints if the hand ends, which
  // sets rewards Otherwise, we should reset rewards since finishing a
  // non-terminal trick awards nothing
  int cards_dealt_before = cards_dealt_;
  MaybeResolveHand(trick_winner);
  bool hand_ended = hand_over_;

  // Don't reset rewards if game became terminal (either hand ended or game
  // ended)
  if (!hand_ended && !terminal_) {
    // Hand continues - reset rewards since finishing a non-terminal trick
    // awards nothing
    std::fill(rewards_.begin(), rewards_.end(), 0.0);

    // Move to next trick
    ++current_trick_index_;
    cards_played_in_trick_ = 0;
    std::fill(played_current_trick_.begin(), played_current_trick_.end(),
              false);

    if (current_trick_index_ < kNumTricks) {
      tricks_[current_trick_index_].cards.clear();
      tricks_[current_trick_index_].players.clear();
    }

    if (!terminal_) {
      if (trick_winner != kTiePlayer) {
        trick_leader_ = trick_winner;
      }
      cur_player_ = trick_leader_;
    }
  }
  // If hand ended, StartNewHand() was called and set cur_player_ =
  // kChancePlayerId to start dealing AwardPoints() set rewards and they need to
  // persist through dealing
}

void TrucoState::MaybeResolveHand(Player latest_trick_winner) {
  if (terminal_) return;
  if (latest_trick_winner != kTiePlayer &&
      trick_wins_[latest_trick_winner] >= 2) {
    winner_ = latest_trick_winner;
  } else if (current_trick_index_ == 0) {
    // Nothing to do after first trick except record potential tie.
  } else if (current_trick_index_ == 1) {
    Player first = trick_results_[0];
    Player second = trick_results_[1];
    // std::cout << "Trick 1 results: first=" << first << " second=" << second
    // << std::endl;
    if (first == kTiePlayer && second != kTiePlayer) {
      winner_ = second;
    } else if (first != kTiePlayer && second == kTiePlayer) {
      winner_ = first;
    } else if (first == kTiePlayer && second == kTiePlayer) {
      // Both first and second tricks tied: the third trick decides.
    }
  } else if (current_trick_index_ == 2) {
    Player third = trick_results_[2];
    if (third != kTiePlayer) {
      winner_ = third;
    } else if (trick_results_[0] != kTiePlayer &&
               trick_results_[0] != kInvalidPlayer) {
      winner_ = trick_results_[0];
    } else if (trick_results_[1] != kTiePlayer &&
               trick_results_[1] != kInvalidPlayer) {
      winner_ = trick_results_[1];
    } else {
      winner_ = mano_;
    }
  }

  if (winner_ != kInvalidPlayer) {
    // Award points for winning the hand (updates game_points_ and returns_)
    AwardPoints(winner_, truco_level_);
    num_hands_played_++;

    // Check if either player reached 30 points to win the game
    if (game_points_[0] >= kTargetScore || game_points_[1] >= kTargetScore) {
      terminal_ = true;
      cur_player_ = kTerminalPlayerId;
    } else {
      // Continue to next hand
      hand_over_ = true;
      cur_player_ = winner_;
    }
  }
}

void TrucoState::StartNewHand() {
  hand_over_ = false;
  // Reset hand-specific state while preserving game_points_
  winner_ = kInvalidPlayer;
  cards_dealt_ = 0;
  cards_played_in_trick_ = 0;
  current_trick_index_ = 0;

  // Alternate mano (starting player rotates each hand)
  mano_ = 1 - mano_;
  trick_leader_ = mano_;

  // Reset hands and card tracking
  for (auto& hand : player_hands_) {
    hand.clear();
  }
  std::fill(card_owner_.begin(), card_owner_.end(), kInvalidPlayer);
  std::fill(card_played_.begin(), card_played_.end(), false);
  std::fill(played_current_trick_.begin(), played_current_trick_.end(), false);
  std::fill(trick_wins_.begin(), trick_wins_.end(), 0);
  std::fill(trick_results_.begin(), trick_results_.end(), kInvalidPlayer);

  // Reset tricks
  for (auto& trick : tricks_) {
    trick.cards.clear();
    trick.players.clear();
  }

  // Reset betting state
  pending_response_ = PendingResponse::kNone;
  player_turn_before_bet_ = kInvalidPlayer;
  envido_sequence_.clear();
  envido_last_caller_ = kInvalidPlayer;
  envido_resolved_ = false;
  envido_locked_ = false;
  truco_level_ = 1;
  pending_truco_target_ = 0;
  pending_truco_caller_ = kInvalidPlayer;
  truco_next_raiser_ = kInvalidPlayer;
  response_stack_.clear();

  // Start dealing phase
  cur_player_ = kChancePlayerId;
}

std::vector<Action> TrucoState::LegalChanceOutcomes() const {
  std::vector<Action> actions;
  for (Action card = 0; card < kNumCards; ++card) {
    if (card_owner_[card] == kInvalidPlayer) {
      actions.push_back(card);
    }
  }
  return actions;
}

Player TrucoState::NextPlayerInTrick(Player after) const {
  for (int offset = 1; offset <= num_players_; ++offset) {
    Player candidate = (after + offset) % num_players_;
    if (!played_current_trick_[candidate]) {
      return candidate;
    }
  }
  return kInvalidPlayer;
}

TrucoGame::TrucoGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      total_cards_(kNumCards),
      starting_player_(ParameterValue<int>("starting_player")),
      deck_(CreateDeck()) {
  SPIEL_CHECK_EQ(num_players_, 2);
  SPIEL_CHECK_GE(starting_player_, 0);
  SPIEL_CHECK_LT(starting_player_, num_players_);
  default_observer_ = std::make_shared<TrucoObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<TrucoObserver>(kInfoStateObsType);
}

std::unique_ptr<State> TrucoGame::NewInitialState() const {
  return absl::make_unique<TrucoState>(shared_from_this());
}

int TrucoGame::MaxChanceOutcomes() const { return total_cards_; }

double TrucoGame::MinUtility() const {
  // Game is played to 30 points, so min utility is -30
  return -static_cast<double>(kTargetScore);
}

double TrucoGame::MaxUtility() const {
  // Game is played to 30 points, so max utility is 30
  return static_cast<double>(kTargetScore);
}

std::vector<int> TrucoGame::InformationStateTensorShape() const {
  int player_bits = num_players_;
  int single_hand_bits = kNumCards;
  int all_hand_bits = num_players_ * kNumCards;
  int trick_bits = kNumTricks * num_players_ * kNumCards;
  int envido_bits = kEnvidoTensorSize;
  int truco_bits = kTrucoTensorSize;
  int pending_bits = kPendingTensorSize;
  int game_score_bits = num_players_;  // Game scores for Falta Envido strategy
  return {player_bits + single_hand_bits + all_hand_bits + trick_bits +
          envido_bits + truco_bits + pending_bits + game_score_bits};
}

std::vector<int> TrucoGame::ObservationTensorShape() const {
  int player_bits = num_players_;
  int hand_bits = kNumCards;
  int all_hand_bits = num_players_ * kNumCards;
  int trick_bits = kNumTricks * num_players_ * kNumCards;
  int envido_bits = kEnvidoTensorSize;
  int truco_bits = kTrucoTensorSize;
  int pending_bits = kPendingTensorSize;
  int game_score_bits = num_players_;  // Game scores for Falta Envido strategy
  int mano_bits = num_players_;        // Mano indicator for tie-breaking
  return {player_bits + hand_bits + all_hand_bits + trick_bits + envido_bits +
          truco_bits + pending_bits + game_score_bits + mano_bits};
}

std::string TrucoGame::ActionToString(Player /*player*/, Action action) const {
  if (IsCardAction(action)) {
    return CardById(action).ToString();
  }
  return SpecialActionToString(action);
}

std::shared_ptr<Observer> TrucoGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  IIGObservationType obs_type = iig_obs_type.value_or(kDefaultObsType);
  if (params.empty()) {
    return std::make_shared<TrucoObserver>(obs_type);
  }
  return std::make_shared<TrucoObserver>(obs_type);
}

}  // namespace truco
}  // namespace open_spiel
