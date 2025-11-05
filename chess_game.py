import json
import chess
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
             'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

class ChessMovePredictor(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc_move = nn.Linear(256, num_moves)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc_move(x)
    
def model_prep(move2idx_path, model_path):
    with open(move2idx_path, 'r') as f:
        move2idx = json.load(f)

    model = ChessMovePredictor(num_moves=len(move2idx))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, move2idx


def fen_to_tensor(fen):
    board = chess.Board(fen)
    mat = np.zeros((8,8,12), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            mat[row, col, piece_map[piece.symbol()]] = 1.0
    turn_channel = np.full((8,8,1), int(board.turn), dtype=np.float32)
    mat = np.concatenate([mat, turn_channel], axis=-1)
    return mat


def choose_move(board, move2idx, probs):
    
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')
    for move in legal_moves:
        move_str = move.uci()
        if move_str in move2idx.keys():
            move_score = probs[move2idx[move_str]]
            if move_score > best_score:
                best_score = move_score
                best_move = move

    if best_move == None:
        best_move = legal_moves[0]

    return best_move


def make_move(board, model, device, move2idx):
    with torch.no_grad():
        board_tensor = torch.tensor(fen_to_tensor(board.fen()), dtype=torch.float32)
        board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        logits_move = model(board_tensor)
        probs = torch.softmax(logits_move, dim=1).cpu().numpy().flatten()

        return choose_move(board,move2idx, probs)


def model_vs_human(board):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, move2idx =  model_prep("move_idx.json","chess_model.pth")
    return make_move(board, model, device, move2idx)


    
def model_vs_model():
    fens, moves = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, move2idx1 =  model_prep("move_idx.json","chess_model.pth")
    model2, move2idx2 =  model_prep("move_idx_old.json","chess_model_old.pth")

    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = make_move(board, model1, device, move2idx1)
        else:
            if random.randint(0,1)==0:
                legal_moves = list(board.legal_moves)
                move = legal_moves[random.randint(0, len(legal_moves)-1)]
            else:
                move = make_move(board, model2, device, move2idx2)
        fens.append(board.fen())
        moves.append(move)
        board.push(move)
    return board.outcome(), fens, moves


def model_game(times=10):
    data = {"fens":[],"moves":[],"score":[]}
    count=0
    while len(data["fens"])<10000:
    # for i in range(times):

        flag = True
        outcome, fens, moves = model_vs_model() 
        score = [1 if i%2==0 else -1 for i in range(len(fens))]
        if outcome.winner == chess.BLACK:
            tmp = score.pop(0)
            score.append(-1*score[-1])
        elif outcome.winner != chess.WHITE:
            flag = False
            # score = [0 for _ in range(len(fens))]
        if flag:
            data["fens"].extend(fens)
            data["moves"].extend(moves)
            data["score"].extend(score)
            print(len(data["fens"]))

    df = pd.DataFrame(data)
    print(len(df))
    df.to_csv("chess_positions_new.csv", index=False)


def main():
    model_game()

if __name__ == "__main__":
    main()
