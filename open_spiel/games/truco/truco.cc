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
#include <array>
#include <functional>
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

class TrucoObserver : public Observer {
 public:
  explicit TrucoObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  static void WriteObservingPlayer(const TrucoState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  static void WritePrivateCards(const TrucoState& state, int player,
                                Allocator* allocator) {
    auto out = allocator->Get("hand", {kNumCards});
    for (int card : state.PlayerHand(player)) {
      if (!state.card_played_[card]) {
        out.at(card) = 1;
      }
    }
  }

  static void WriteAllPrivateCards(const TrucoState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("hands", {state.num_players_, kNumCards});
    for (Player p = 0; p < state.num_players_; ++p) {
      for (int card : state.PlayerHand(p)) {
        out.at(p, card) = 1;
      }
    }
  }

  static void WriteTrickHistory(const TrucoState& state, Allocator* allocator) {
    auto out =
        allocator->Get("tricks", {kNumTricks, state.num_players_, kNumCards});
    for (int trick = 0; trick < kNumTricks; ++trick) {
      const Trick& t = state.trick_history()[trick];
      for (int idx = 0; idx < t.cards.size(); ++idx) {
        // Use player_id for indexing to make it invariant to turn order
        Player p = t.players[idx];
        out.at(trick, p, t.cards[idx]) = 1;
      }
    }
  }

  static void WriteTrickAnalysis(const TrucoState& state,
                                 Allocator* allocator) {
    // Winners: P0, P1, Tie (3 bits per trick)
    // Leaders: P0, P1 (2 bits per trick)
    auto out_winners =
        allocator->Get("trick_winners", {kNumTricks, state.num_players_ + 1});
    auto out_leaders =
        allocator->Get("trick_leaders", {kNumTricks, state.num_players_});

    for (int t = 0; t < kNumTricks; ++t) {
      // Winner
      Player winner = state.trick_results()[t];
      if (winner != kInvalidPlayer) {
        if (winner == kTiePlayer) {
          out_winners.at(t, state.num_players_) = 1;  // Tie
        } else {
          out_winners.at(t, winner) = 1;
        }
      }

      // Leader
      if (t < state.trick_history().size() &&
          !state.trick_history()[t].players.empty()) {
        Player leader = state.trick_history()[t].players[0];
        out_leaders.at(t, leader) = 1;
      }
    }
  }

  static void WriteCurrentTrucoLevel(const TrucoState& state,
                                     Allocator* allocator) {
    auto out = allocator->Get("truco_level", {kTrucoLevelBits});
    if (state.truco_level_ >= 1 && state.truco_level_ <= 4) {
      out.at(state.truco_level_ - 1) = 1;
    }
  }

  static void WriteEnvidoSequence(const TrucoState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("envido_sequence",
                              {kMaxEnvidoSequenceActions, kNumEnvidoTypes});
    for (size_t i = 0;
         i < state.envido_sequence_.size() && i < kMaxEnvidoSequenceActions;
         ++i) {
      out.at(i, static_cast<int>(state.envido_sequence_[i])) = 1;
    }
  }

  static void WriteEnvidoState(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("envido_state", {kEnvidoStateBits});
    if (state.envido_resolved_) {
      out.at(0) = 1;
    }
    if (state.envido_locked_) {
      out.at(1) = 1;
    }
  }

  static void WriteMano(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("mano", {state.num_players_});
    out.at(state.mano_) = 1;
  }

  static void WriteEnvidoScores(const TrucoState& state, int player,
                                Allocator* allocator) {
    auto out = allocator->Get("envido_scores", {state.num_players_});

    // Always write own Envido score (helps agent learn Envido calculation)
    out.at(player) = static_cast<float>(state.PlayerEnvidoScore(player)) / 33.0;

    // Write opponent's score only if it was revealed (Envido accepted)
    if (state.revealed_envido_scores_[1 - player] >= 0) {
      Player opponent = 1 - player;
      out.at(opponent) =
          static_cast<float>(state.revealed_envido_scores_[opponent]) / 33.0;
    }
  }

  static void WriteGameScores(const TrucoState& state, Allocator* allocator) {
    auto out = allocator->Get("game_scores", {state.num_players_});
    for (Player p = 0; p < state.num_players_; ++p) {
      out.at(p) = static_cast<float>(state.game_points_[p]) / kTargetScore;
    }
  }

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    const auto& state =
        open_spiel::down_cast<const TrucoState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    WriteObservingPlayer(state, player, allocator);

    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WritePrivateCards(state, player, allocator);
    }

    if (iig_obs_type_.public_info) {
      WriteTrickHistory(state, allocator);
      WriteTrickAnalysis(state, allocator);
      WriteGameScores(state, allocator);
      WriteCurrentTrucoLevel(state, allocator);
      WriteEnvidoSequence(state, allocator);
      WriteEnvidoState(state, allocator);
      WriteEnvidoScores(state, player, allocator);
      WriteMano(state, allocator);
    }
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    const auto& state =
        open_spiel::down_cast<const TrucoState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    std::string result = absl::StrCat("[Observer:", player, "]");

    if (iig_obs_type_.private_info != PrivateInfoType::kNone) {
      std::vector<std::string> cards;
      const TrucoGame& game = state.ParentGame();
      for (int card : state.PlayerHand(player)) {
        cards.push_back(CardString(game, card));
      }
      absl::StrAppend(&result, "[Hand:", absl::StrJoin(cards, ","), "]");
    }

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
      absl::StrAppend(&result, "[GameScore:", state.game_points_[0], ",",
                      state.game_points_[1], "]");
    }

    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

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

const TrucoGame& TrucoState::ParentGame() const {
  return open_spiel::down_cast<const TrucoGame&>(*game_);
}

TrucoState::TrucoState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId),
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

  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");
  absl::StrAppend(&result, "| Truco - Hand ", num_hands_played_ + 1, "\n");
  absl::StrAppend(&result, "| Score: P0 [ ", game_points_[0], " ]  vs  P1 [ ",
                  game_points_[1], " ]\n");
  absl::StrAppend(&result, "| Mano: P", mano_, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

  absl::StrAppend(&result, "| P1 Hand: ");
  for (int card : player_hands_[1]) {
    if (!card_played_[card]) {
      absl::StrAppend(&result, "[", CardString(game, card), "] ");
    }
  }
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

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

  absl::StrAppend(&result, "| P0 Hand: ");
  for (int card : player_hands_[0]) {
    if (!card_played_[card]) {
      absl::StrAppend(&result, "[", CardString(game, card), "] ");
    }
  }
  absl::StrAppend(&result, "\n");
  absl::StrAppend(&result,
                  "+--------------------------------------------------+\n");

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

  if (IsTerminal()) {
    absl::StrAppend(&result, "| GAME OVER\n");
    absl::StrAppend(&result, "| Final Score: P0=", game_points_[0],
                    "  P1=", game_points_[1], "\n");
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

std::vector<double> TrucoState::Returns() const { return returns_; }

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

const std::vector<int>& TrucoState::PlayerHand(Player player) const {
  return player_hands_[player];
}

std::unique_ptr<State> TrucoState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = Clone();
  auto* truco_clone = open_spiel::down_cast<TrucoState*>(clone.get());

  Player opponent = 1 - player_id;
  std::vector<int> available_cards;
  std::vector<int> opponent_slots;
  std::vector<int> opponent_played_cards;

  // Identify opponent's unplayed card slots and played cards
  for (size_t i = 0; i < truco_clone->player_hands_[opponent].size(); ++i) {
    int card = truco_clone->player_hands_[opponent][i];
    if (!truco_clone->card_played_[card]) {
      opponent_slots.push_back(i);
    } else {
      opponent_played_cards.push_back(card);
    }
  }

  // Collect all available cards (unowned + opponent's current unplayed cards)
  for (int card = 0; card < kNumCards; ++card) {
    if (truco_clone->card_owner_[card] == kInvalidPlayer ||
        (truco_clone->card_owner_[card] == opponent &&
         !truco_clone->card_played_[card])) {
      available_cards.push_back(card);
    }
  }

  int num_cards_needed = opponent_slots.size();
  uint32_t seed =
      static_cast<uint32_t>(rng() * std::numeric_limits<uint32_t>::max());
  std::mt19937 gen(seed);

  // If Envido was resolved, we must maintain the opponent's revealed score
  // The score is calculated from their FULL 3-card hand (played + unplayed)
  if (envido_resolved_ && revealed_envido_scores_[opponent] >= 0) {
    int required_score = revealed_envido_scores_[opponent];

    // Find card combinations where: played_cards + new_cards = required_score
    std::vector<std::vector<int>> valid_combinations =
        FindCardCombinationsWithEnvidoScoreGivenFixed(
            available_cards, num_cards_needed, opponent_played_cards,
            required_score);

    if (valid_combinations.empty()) {
      SpielFatalError(absl::StrCat(
          "No valid card combination exists for Envido score: ", required_score,
          " with ", opponent_played_cards.size(), " fixed cards and ",
          num_cards_needed, " cards from ", available_cards.size(),
          " available cards"));
    }

    // Sample uniformly from valid combinations
    std::uniform_int_distribution<size_t> dist(0,
                                               valid_combinations.size() - 1);
    const std::vector<int>& chosen_combination = valid_combinations[dist(gen)];

    // Assign chosen cards to opponent's hand
    for (size_t i = 0; i < opponent_slots.size(); ++i) {
      int slot = opponent_slots[i];
      int new_card = chosen_combination[i];
      truco_clone->player_hands_[opponent][slot] = new_card;
      truco_clone->card_owner_[new_card] = opponent;
    }

    // Mark remaining cards as unowned
    std::set<int> assigned(chosen_combination.begin(),
                           chosen_combination.end());
    for (int card : available_cards) {
      if (assigned.find(card) == assigned.end()) {
        truco_clone->card_owner_[card] = kInvalidPlayer;
      }
    }

    return clone;
  }

  // No Envido constraint - do simple shuffle
  std::shuffle(available_cards.begin(), available_cards.end(), gen);

  for (size_t i = 0; i < opponent_slots.size(); ++i) {
    int slot = opponent_slots[i];
    int new_card = available_cards[i];
    truco_clone->player_hands_[opponent][slot] = new_card;
    truco_clone->card_owner_[new_card] = opponent;
  }

  for (size_t i = opponent_slots.size(); i < available_cards.size(); ++i) {
    truco_clone->card_owner_[available_cards[i]] = kInvalidPlayer;
  }

  return clone;
}

void TrucoState::DoApplyAction(Action move) {
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  if (hand_over_) {
    SPIEL_CHECK_EQ(move, kNewHandAction);
    StartNewHand();
    return;
  }

  if (cur_player_ == kChancePlayerId) {
    DealCard(move);
  } else {
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
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  EnvidoCall call;
  if (action == kEnvidoAction) {
    call = EnvidoCall::kEnvido;
  } else if (action == kRealEnvidoAction) {
    call = EnvidoCall::kRealEnvido;
  } else {
    call = EnvidoCall::kFaltaEnvido;
  }

  Player caller = cur_player_;

  if (pending_response_ == PendingResponse::kTruco) {
    response_stack_.push_back({pending_response_, pending_truco_caller_,
                               pending_truco_target_, cur_player_});
    pending_response_ = PendingResponse::kEnvido;
    envido_last_caller_ = cur_player_;
    cur_player_ = Opponent(cur_player_);
  } else if (pending_response_ == PendingResponse::kNone) {
    BeginPendingResponse(PendingResponse::kEnvido, caller);
    envido_last_caller_ = caller;
  } else {
    SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
    envido_last_caller_ = cur_player_;
    cur_player_ = Opponent(cur_player_);
  }
  envido_sequence_.push_back(call);
  envido_log_.push_back({caller, call});
}

void TrucoState::ApplyTrucoCall(Action action) {
  std::fill(rewards_.begin(), rewards_.end(), 0.0);
  SPIEL_CHECK_EQ(action, kRaiseTrucoAction);

  int target = 0;
  if (pending_response_ == PendingResponse::kTruco) {
    target = pending_truco_target_ + 1;
  } else {
    target = truco_level_ + 1;
  }
  SPIEL_CHECK_LE(target, 4);

  truco_log_.push_back({cur_player_, target});

  if (pending_response_ == PendingResponse::kTruco) {
    SPIEL_CHECK_GT(pending_truco_target_, 0);
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
    player_turn_before_bet_ = caller;
  }
  cur_player_ = Opponent(caller);
}

void TrucoState::RestoreTurnAfterBet() {
  SPIEL_CHECK_NE(player_turn_before_bet_, kInvalidPlayer);
  cur_player_ = player_turn_before_bet_;
  player_turn_before_bet_ = kInvalidPlayer;
}

void TrucoState::ResolveEnvidoAcceptance() {
  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kEnvido);
  revealed_envido_scores_[0] = PlayerEnvidoScore(0);
  revealed_envido_scores_[1] = PlayerEnvidoScore(1);

  Player winner = DetermineEnvidoWinner();
  int points = 0;
  if (!envido_sequence_.empty() &&
      envido_sequence_.back() == EnvidoCall::kFaltaEnvido) {
    Player loser = Opponent(winner);
    points = FaltaEnvidoValue(loser);
  } else {
    points = SumEnvidoPoints(/*include_last=*/true);
  }
  AwardPoints(winner, points);

  envido_resolved_ = true;
  envido_locked_ = true;
  envido_sequence_.clear();
  envido_log_.clear();
  envido_last_caller_ = kInvalidPlayer;

  if (!response_stack_.empty()) {
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
  envido_log_.clear();
  envido_last_caller_ = kInvalidPlayer;

  if (!response_stack_.empty()) {
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
  std::fill(rewards_.begin(), rewards_.end(), 0.0);

  SPIEL_CHECK_EQ(pending_response_, PendingResponse::kTruco);
  SPIEL_CHECK_GT(pending_truco_target_, 0);
  truco_level_ = pending_truco_target_;
  pending_truco_target_ = 0;
  pending_truco_caller_ = kInvalidPlayer;
  pending_response_ = PendingResponse::kNone;

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

  winner_ = hand_winner;
  num_hands_played_++;

  if (game_points_[0] >= kTargetScore || game_points_[1] >= kTargetScore) {
    terminal_ = true;
    cur_player_ = kTerminalPlayerId;
  } else {
    hand_over_ = true;
    cur_player_ = winner_;
  }
}

void TrucoState::AwardPoints(Player player, int points) {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_GE(points, 0);
  if (points == 0) return;

  int old_score = game_points_[player];
  int new_score = std::min(old_score + points, kTargetScore);
  int actual_points_gained = new_score - old_score;

  game_points_[player] = new_score;

  if (game_points_[player] >= kTargetScore) {
    terminal_ = true;
    cur_player_ = kTerminalPlayerId;
  }

  std::fill(rewards_.begin(), rewards_.end(), 0.0);
  rewards_[player] = static_cast<double>(actual_points_gained);
  rewards_[Opponent(player)] = -static_cast<double>(actual_points_gained);

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
  int leader_score = std::max(game_points_[0], game_points_[1]);
  return std::max(0, kTargetScore - leader_score);
}

int TrucoState::FaltaEnvidoValue(Player /*loser*/) const {
  return FaltaEnvidoValue();
}

int TrucoState::ComputeEnvidoScore(const std::vector<int>& cards) const {
  const TrucoGame& game = ParentGame();
  std::array<std::vector<int>, kNumSuits> suit_values;
  for (int card : cards) {
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

std::vector<std::vector<int>> TrucoState::FindCardCombinationsWithEnvidoScore(
    const std::vector<int>& available_cards, int num_cards,
    int target_score) const {
  std::vector<std::vector<int>> result;
  std::vector<int> current_combination;

  // Helper function to generate combinations recursively
  std::function<void(size_t, int)> generate_combinations = [&](size_t start_idx,
                                                               int remaining) {
    if (remaining == 0) {
      if (ComputeEnvidoScore(current_combination) == target_score) {
        result.push_back(current_combination);
      }
      return;
    }

    if (start_idx + remaining > available_cards.size()) {
      return;
    }

    for (size_t i = start_idx; i < available_cards.size(); ++i) {
      current_combination.push_back(available_cards[i]);
      generate_combinations(i + 1, remaining - 1);
      current_combination.pop_back();
    }
  };

  generate_combinations(0, num_cards);
  return result;
}

std::vector<std::vector<int>>
TrucoState::FindCardCombinationsWithEnvidoScoreGivenFixed(
    const std::vector<int>& available_cards, int num_cards,
    const std::vector<int>& fixed_cards, int target_score) const {
  std::vector<std::vector<int>> result;
  std::vector<int> current_combination;

  std::function<void(size_t, int)> generate_combinations = [&](size_t start_idx,
                                                               int remaining) {
    if (remaining == 0) {
      std::vector<int> full_hand = fixed_cards;
      full_hand.insert(full_hand.end(), current_combination.begin(),
                       current_combination.end());

      if (ComputeEnvidoScore(full_hand) == target_score) {
        result.push_back(current_combination);
      }
      return;
    }

    if (start_idx + remaining > available_cards.size()) {
      return;
    }

    for (size_t i = start_idx; i < available_cards.size(); ++i) {
      current_combination.push_back(available_cards[i]);
      generate_combinations(i + 1, remaining - 1);
      current_combination.pop_back();
    }
  };

  generate_combinations(0, num_cards);
  return result;
}

int TrucoState::PlayerEnvidoScore(Player player) const {
  return ComputeEnvidoScore(player_hands_[player]);
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

  MaybeResolveHand(trick_winner);

  if (!hand_over_ && !terminal_) {
    std::fill(rewards_.begin(), rewards_.end(), 0.0);

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
}

void TrucoState::MaybeResolveHand(Player latest_trick_winner) {
  if (terminal_) return;
  if (latest_trick_winner != kTiePlayer &&
      trick_wins_[latest_trick_winner] >= 2) {
    winner_ = latest_trick_winner;
  } else if (current_trick_index_ == 0) {
    // No winner possible after 1st trick
  } else if (current_trick_index_ == 1) {
    Player first = trick_results_[0];
    Player second = trick_results_[1];
    if (first == kTiePlayer && second != kTiePlayer) {
      winner_ = second;
    } else if (first != kTiePlayer && second == kTiePlayer) {
      winner_ = first;
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
    AwardPoints(winner_, truco_level_);
    num_hands_played_++;

    if (game_points_[0] >= kTargetScore || game_points_[1] >= kTargetScore) {
      terminal_ = true;
      cur_player_ = kTerminalPlayerId;
    } else {
      hand_over_ = true;
      cur_player_ = winner_;
    }
  }
}

void TrucoState::StartNewHand() {
  hand_over_ = false;
  winner_ = kInvalidPlayer;
  cards_dealt_ = 0;
  cards_played_in_trick_ = 0;
  current_trick_index_ = 0;

  mano_ = 1 - mano_;
  trick_leader_ = mano_;

  for (auto& hand : player_hands_) {
    hand.clear();
  }
  std::fill(card_owner_.begin(), card_owner_.end(), kInvalidPlayer);
  std::fill(card_played_.begin(), card_played_.end(), false);
  std::fill(played_current_trick_.begin(), played_current_trick_.end(), false);
  std::fill(trick_wins_.begin(), trick_wins_.end(), 0);
  std::fill(trick_results_.begin(), trick_results_.end(), kInvalidPlayer);

  for (auto& trick : tricks_) {
    trick.cards.clear();
    trick.players.clear();
  }

  pending_response_ = PendingResponse::kNone;
  player_turn_before_bet_ = kInvalidPlayer;
  envido_sequence_.clear();
  envido_log_.clear();
  envido_last_caller_ = kInvalidPlayer;
  envido_resolved_ = false;
  envido_locked_ = false;
  revealed_envido_scores_ = {-1, -1};
  truco_level_ = 1;
  pending_truco_target_ = 0;
  pending_truco_caller_ = kInvalidPlayer;
  truco_next_raiser_ = kInvalidPlayer;
  truco_log_.clear();
  response_stack_.clear();

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
  int trick_info_bits =
      kNumTricks * (num_players_ + 1) + kNumTricks * num_players_;
  int game_score_bits = num_players_;
  int current_truco_level_bits = kTrucoLevelBits;
  int envido_sequence_bits = kEnvidoSequenceTensorSize;
  int envido_state_bits = kEnvidoStateBits;
  int envido_score_bits = num_players_;
  int mano_bits = num_players_;

  return {player_bits + single_hand_bits + all_hand_bits + trick_bits +
          trick_info_bits + game_score_bits + current_truco_level_bits +
          envido_sequence_bits + envido_state_bits + envido_score_bits +
          mano_bits};
}

std::vector<int> TrucoGame::ObservationTensorShape() const {
  int player_bits = num_players_;
  int hand_bits = kNumCards;
  int trick_bits = kNumTricks * num_players_ * kNumCards;
  int trick_info_bits =
      kNumTricks * (num_players_ + 1) + kNumTricks * num_players_;
  int game_score_bits = num_players_;
  int current_truco_level_bits = kTrucoLevelBits;
  int envido_sequence_bits = kEnvidoSequenceTensorSize;
  int envido_state_bits = kEnvidoStateBits;
  int envido_score_bits = num_players_;
  int mano_bits = num_players_;

  return {player_bits + hand_bits + trick_bits + trick_info_bits +
          game_score_bits + current_truco_level_bits + envido_sequence_bits +
          envido_state_bits + envido_score_bits + mano_bits};
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
