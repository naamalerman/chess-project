import chess
import numpy as np
import pandas as pd

import torch

piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
             'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

piece_weight = {'P':0.1,'N':0.15,'B':0.15,'R':0.2,'Q':0.25,'K':0.0,
               'p':0.1,'n':0.15,'b':0.15,'r':0.2,'q':0.25,'k':0.0}


def move_weight(board, move):
    piece = board.piece_at(move.from_square)
    piece_to = board.piece_at(move.to_square)

    value_from = piece_weight[piece.symbol()]
    value_to = piece_weight[piece_to.symbol()] if piece_to else 0
    return value_to - value_from


def midgame_move_weight(board, move):
    move_score = 0
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)

    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    if move.to_square in center_squares:
        move_score += 0.5

    if moving_piece and moving_piece.symbol().lower() in ['n', 'b']:
        rank = chess.square_rank(move.from_square)
        if (moving_piece.color and rank == 0) or (not moving_piece.color and rank == 7):
            move_score += 1.0

    piece_type = moving_piece.piece_type if moving_piece else None
    if piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        move_score += 0.1

    if board.is_castling(move):
        move_score += 1.0

    if captured_piece and moving_piece:
        gain = piece_weight[captured_piece.symbol()] - piece_weight[moving_piece.symbol()]
        move_score += 0.7 * gain

    return move_score

def endgame_move_weight(board, move):
    move_score = 0
    moving_piece = board.piece_at(move.from_square)

    if moving_piece and moving_piece.symbol().lower() == 'p':
        rank = chess.square_rank(move.to_square)
        if (moving_piece.color and rank == 7) or (not moving_piece.color and rank == 0):
            move_score += 2.0
        elif (moving_piece.color and rank >= 5) or (not moving_piece.color and rank <= 2):
            move_score += 1.0

    board.push(move)
    if board.is_check():
        move_score += 1.2
        if not board.is_checkmate() and not board.is_attacked_by(not moving_piece.color, board.king(moving_piece.color)):
            move_score += 0.5
    board.pop()

    if moving_piece and moving_piece.symbol().lower() == 'k':
        move_score -= 0.8
    else:
        move_score += 0.2 * piece_weight[moving_piece.symbol()]

    return move_score

def get_eval(board, model, board_tensor, move):
    board.push(move)
    
    with torch.no_grad():
        _, pred_eval = model(board_tensor)
        move_eval_score = float(pred_eval.item())

    board.pop()
    return move_eval_score

def capture_and_risk(board, move):
    moving_piece = board.piece_at(move.from_square)
    captured_piece = board.piece_at(move.to_square)
    
    capture_score = 0
    if captured_piece:
        gain = piece_weight[captured_piece.symbol()] - piece_weight[moving_piece.symbol()]
        if moving_piece.symbol().lower() == 'k':
            gain = piece_weight[captured_piece.symbol()]*0.5
        capture_score = 3.0 * max(gain, 0.05)
        if not board.is_attacked_by(not moving_piece.color, move.to_square):
            capture_score += 1.0
    
    risk_score = 0
    if board.is_attacked_by(not moving_piece.color, move.to_square) and moving_piece.symbol().lower() in ['q','r','k']:
        risk_score -= 1.2

    return capture_score + risk_score
    

def calc_weight(board, model, board_tensor, move2idx, move, probs, type=1):
    prob = 0
    endgame_factor = 1 - (len(board.piece_map()) / 32)
    move_str = move.uci()

    move_eval_score = get_eval(board, model, board_tensor, move) if type==2 else 0
    prob = probs[move2idx[move_str]] if move_str in move2idx else 0
    capture_risk_score = capture_and_risk(board, move)
    
    development_score = midgame_move_weight(board, move) * (1 - endgame_factor)
    endgame_score = endgame_move_weight(board, move) * endgame_factor
    if endgame_factor>0.5:
        return 0.2 * (prob + move_eval_score) + capture_risk_score*0.5 #+ endgame_move_weight(board, move) * endgame_factor 

    return 0.5 * prob + 0.7 * move_eval_score + capture_risk_score *0.5
        # + 0.3 * piece_weight[board.piece_at(move.from_square).symbol()] + development_score + endgame_score
    

    
    game_factors = 0.07 * (endgame_factor * endgame_move_weight(board, move) 
                          + (1 - endgame_factor) * midgame_move_weight(board, move))

    return 0.6 * prob + 0.5 * move_eval_score + 0.5 * piece_weight[board.piece_at(move.from_square).symbol()] + game_factors 



            
