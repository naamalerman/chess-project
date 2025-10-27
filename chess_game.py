import json
import chess
import numpy as np

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
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc_move = nn.Linear(256, num_moves)
        self.fc_eval = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc_move(x), self.fc_eval(x)
    
def model_prep():
    with open('move_idx.json', 'r') as f:
        move2idx = json.load(f)
    idx2move = {i:m for m,i in move2idx.items()}

    model = ChessMovePredictor(num_moves=len(move2idx))
    model.load_state_dict(torch.load("chess_model.pth", map_location="cpu"))
    model.eval()
    return model, idx2move

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
    best_move = legal_moves[0]
    best_score = -float('inf')

    for move in legal_moves:
        move_str = str(move)

        if move_str in move2idx:
            move_score = probs[move2idx[move_str]]
            if move_score > best_score:
                best_score = move_score
                best_move = move
    
    return best_move


def model_move(board):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, move2idx =  model_prep()

    with torch.no_grad():
        temp_board = board.copy()

        board_tensor = torch.tensor(fen_to_tensor(temp_board.fen()), dtype=torch.float32)
        board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        logits_move, pred_eval = model(board_tensor)
        probs = torch.softmax(logits_move, dim=1).cpu().numpy().flatten()

        return choose_move(board,move2idx, probs)
        
